#!/usr/bin/env python
"""
词汇表验证脚本：彻底解决CausalQwen词汇表大小问题

本脚本专门验证以下问题：
1. Qwen原始词汇表大小 K
2. CausalQwen词汇表大小 K+1
3. <NUM> token的正确添加和ID获取
4. 分词器行为的完整验证
5. 模型权重形状对比验证

目标：让用户清清楚楚、明明白白地理解词汇表设计
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.tokenizer import QwenTokenizerWrapper
from src.models.causal_lm import CausalLMConfig, CausalLanguageModel


def print_separator(title, level=1):
    """打印分隔符"""
    if level == 1:
        print("\n" + "=" * 80)
        print(f"=   {title.center(74)}   =")
        print("=" * 80)
    else:
        print("\n" + "-" * 60)
        print(f" {title.center(56)} ")
        print("-" * 60)


def verify_original_qwen_tokenizer(model_path):
    """验证原始Qwen分词器的词汇表大小"""
    print_separator("步骤1：验证原始Qwen分词器", 2)
    
    try:
        # 直接加载Qwen分词器
        expanded_path = os.path.expanduser(model_path)
        print(f"正在加载原始Qwen分词器: {expanded_path}")
        
        original_tokenizer = AutoTokenizer.from_pretrained(expanded_path, trust_remote_code=True)
        
        # 获取原始词汇表信息
        original_vocab_size = len(original_tokenizer)
        original_vocab = original_tokenizer.get_vocab()
        
        print(f"✅ 原始Qwen分词器加载成功")
        print(f"   词汇表大小 K = {original_vocab_size}")
        print(f"   pad_token_id = {original_tokenizer.pad_token_id}")
        print(f"   eos_token_id = {original_tokenizer.eos_token_id}")
        print(f"   bos_token_id = {original_tokenizer.bos_token_id}")
        
        # 检查是否已经有<NUM> token
        num_token = "<NUM>"
        has_num_token = num_token in original_vocab
        
        print(f"\n<NUM> token 检查:")
        print(f"   原始词汇表中是否有'{num_token}': {'是' if has_num_token else '否'}")
        
        if has_num_token:
            num_token_id = original_vocab[num_token]
            print(f"   原始'{num_token}' token ID: {num_token_id}")
        else:
            print(f"   原始词汇表中没有'{num_token}' token (符合预期)")
        
        return original_tokenizer, original_vocab_size, has_num_token
        
    except Exception as e:
        print(f"❌ 原始Qwen分词器加载失败: {e}")
        return None, 0, False


def verify_causal_qwen_tokenizer(model_path):
    """验证CausalQwen分词器的词汇表大小"""
    print_separator("步骤2：验证CausalQwen分词器", 2)
    
    try:
        # 使用我们的包装器
        print(f"正在初始化QwenTokenizerWrapper...")
        
        causal_tokenizer = QwenTokenizerWrapper(
            model_path=model_path,
            use_real_tokenizer=True
        )
        
        # 获取CausalQwen词汇表信息
        causal_vocab_size = causal_tokenizer.vocab_size
        causal_num_token_id = causal_tokenizer.num_token_id
        
        print(f"✅ CausalQwen分词器初始化成功")
        print(f"   词汇表大小 = {causal_vocab_size}")
        print(f"   <NUM> token ID = {causal_num_token_id}")
        
        # 验证<NUM> token的功能
        print(f"\n<NUM> token 功能验证:")
        
        # 测试convert_ids_to_tokens
        try:
            num_token_str = causal_tokenizer.convert_ids_to_tokens([causal_num_token_id])[0]
            print(f"   ID {causal_num_token_id} -> token: '{num_token_str}'")
            
            # 测试convert_tokens_to_ids (如果有这个方法)
            if hasattr(causal_tokenizer.tokenizer, 'convert_tokens_to_ids'):
                back_to_id = causal_tokenizer.tokenizer.convert_tokens_to_ids('<NUM>')
                print(f"   token '<NUM>' -> ID: {back_to_id}")
                
                id_consistency = (back_to_id == causal_num_token_id)
                print(f"   ID往返一致性: {'✅' if id_consistency else '❌'}")
        except Exception as e:
            print(f"   ⚠️ token转换测试失败: {e}")
        
        return causal_tokenizer, causal_vocab_size, causal_num_token_id
        
    except Exception as e:
        print(f"❌ CausalQwen分词器初始化失败: {e}")
        return None, 0, -1


def verify_model_architectures(model_path, causal_tokenizer):
    """验证Qwen和CausalQwen模型架构对比"""
    print_separator("步骤3：模型架构权重形状验证", 2)
    
    try:
        # 加载原始Qwen模型
        print(f"正在加载原始Qwen模型...")
        expanded_path = os.path.expanduser(model_path)
        qwen_model = AutoModelForCausalLM.from_pretrained(
            expanded_path,
            torch_dtype=torch.float32,
            device_map=None
        )
        
        # 获取Qwen模型信息
        qwen_config = qwen_model.config
        qwen_hidden_size = qwen_config.hidden_size
        qwen_lm_head = qwen_model.lm_head
        qwen_lm_head_weight_shape = qwen_lm_head.weight.shape
        
        print(f"✅ 原始Qwen模型加载成功")
        print(f"   隐藏层大小 d_model = {qwen_hidden_size}")
        print(f"   lm_head权重形状 = {qwen_lm_head_weight_shape}")
        print(f"   预期lm_head形状 = [K, d_model] = [{len(causal_tokenizer.tokenizer)-1}, {qwen_hidden_size}]")
        
        # 验证Qwen lm_head形状
        expected_qwen_shape = (len(causal_tokenizer.tokenizer)-1, qwen_hidden_size)  # K, d_model
        qwen_shape_correct = (qwen_lm_head_weight_shape == expected_qwen_shape)
        print(f"   Qwen lm_head形状验证: {'✅ 正确' if qwen_shape_correct else '❌ 错误'}")
        
        # 创建CausalQwen模型
        print(f"\n正在创建CausalQwen模型...")
        causal_config = CausalLMConfig(
            vocab_size=causal_tokenizer.vocab_size,
            num_token_id=causal_tokenizer.num_token_id,
            hidden_size=qwen_hidden_size,  # 使用相同的隐藏层大小
            causal_dim=qwen_hidden_size,   # C=H约束
            use_real_qwen=True,
            qwen_model_path=model_path,
            ovr_threshold=10.0,
            reg_loss_weight=1.0
        )
        
        causal_model = CausalLanguageModel(causal_config)
        
        # 获取CausalQwen分类头信息
        causal_cls_head = causal_model.action_network.classification_head.causal_linear
        causal_cls_head_weight_shape = causal_cls_head.weight.shape
        causal_hidden_size = causal_config.hidden_size
        
        print(f"✅ CausalQwen模型创建成功")
        print(f"   隐藏层大小 d_model = {causal_hidden_size}")
        print(f"   分类头权重形状 = {causal_cls_head_weight_shape}")
        print(f"   预期分类头形状 = [K+1, d_model] = [{causal_tokenizer.vocab_size}, {causal_hidden_size}]")
        
        # 验证CausalQwen分类头形状
        expected_causal_shape = (causal_tokenizer.vocab_size, causal_hidden_size)  # K+1, d_model
        causal_shape_correct = (causal_cls_head_weight_shape == expected_causal_shape)
        print(f"   CausalQwen分类头形状验证: {'✅ 正确' if causal_shape_correct else '❌ 错误'}")
        
        # 形状对比总结
        print(f"\n📊 模型权重形状对比总结:")
        print(f"   Qwen lm_head:      {qwen_lm_head_weight_shape}")
        print(f"   CausalQwen 分类头: {causal_cls_head_weight_shape}")
        print(f"   词汇表大小差异:   {causal_cls_head_weight_shape[0] - qwen_lm_head_weight_shape[0]} (应该=1)")
        print(f"   隐藏层大小一致:   {'✅' if qwen_lm_head_weight_shape[1] == causal_cls_head_weight_shape[1] else '❌'}")
        
        # 计算参数数量对比
        qwen_lm_head_params = qwen_lm_head_weight_shape[0] * qwen_lm_head_weight_shape[1]
        causal_cls_head_params = causal_cls_head_weight_shape[0] * causal_cls_head_weight_shape[1]
        param_diff = causal_cls_head_params - qwen_lm_head_params
        
        print(f"\n🔢 参数数量对比:")
        print(f"   Qwen lm_head参数:      {qwen_lm_head_params:,}")
        print(f"   CausalQwen分类头参数:  {causal_cls_head_params:,}")
        print(f"   参数增加量:            {param_diff:,}")
        print(f"   增加率:                {param_diff/qwen_lm_head_params*100:.4f}%")
        
        return qwen_model, causal_model, qwen_shape_correct and causal_shape_correct
        
    except Exception as e:
        print(f"❌ 模型架构验证失败: {e}")
        return None, None, False


def verify_weight_inheritance(qwen_model, causal_model, causal_tokenizer):
    """验证权重继承"""
    print_separator("步骤4：权重继承验证", 2)
    
    try:
        # 执行知识传输初始化
        print(f"正在执行知识传输初始化...")
        num_target_median = 50.0
        num_target_scale = 25.0
        causal_model.init_weights(num_target_median, num_target_scale)
        
        # 获取权重
        qwen_lm_head_weight = qwen_model.lm_head.weight.data
        causal_cls_head_weight = causal_model.action_network.classification_head.causal_linear.weight.data
        
        # 检查前K个token的权重继承
        K = qwen_lm_head_weight.shape[0]  # 原始Qwen词汇表大小
        inherited_weight = causal_cls_head_weight[:K, :]  # 前K行应该继承自Qwen
        
        print(f"✅ 知识传输初始化完成")
        print(f"   检查前{K}个token的权重继承...")
        
        # 计算权重相似度
        weight_diff = torch.abs(inherited_weight - qwen_lm_head_weight)
        max_diff = weight_diff.max().item()
        mean_diff = weight_diff.mean().item()
        
        print(f"   权重差异统计:")
        print(f"     最大差异: {max_diff:.6f}")
        print(f"     平均差异: {mean_diff:.6f}")
        
        # 检查权重是否完全一致
        weights_identical = torch.allclose(inherited_weight, qwen_lm_head_weight, atol=1e-6)
        print(f"   权重完全一致: {'✅' if weights_identical else '❌'}")
        
        # 检查<NUM> token的权重初始化
        num_token_id = causal_tokenizer.num_token_id
        num_token_weight = causal_cls_head_weight[num_token_id, :]
        
        print(f"\n<NUM> token (ID: {num_token_id}) 权重分析:")
        print(f"   权重统计: 均值={num_token_weight.mean().item():.6f}, 标准差={num_token_weight.std().item():.6f}")
        print(f"   权重范围: [{num_token_weight.min().item():.6f}, {num_token_weight.max().item():.6f}]")
        
        # 与继承权重对比
        inherited_weight_mean = inherited_weight.mean(dim=0)
        cosine_sim = torch.nn.functional.cosine_similarity(num_token_weight, inherited_weight_mean, dim=0).item()
        print(f"   与继承权重余弦相似度: {cosine_sim:.6f}")
        
        return weights_identical
        
    except Exception as e:
        print(f"❌ 权重继承验证失败: {e}")
        return False


def compare_vocabularies(original_tokenizer, original_vocab_size, causal_tokenizer, causal_vocab_size):
    """对比两个分词器的词汇表"""
    print_separator("步骤5：词汇表大小对比验证", 2)
    
    expected_causal_size = original_vocab_size + 1
    size_correct = (causal_vocab_size == expected_causal_size)
    
    print(f"词汇表大小对比:")
    print(f"   原始Qwen词汇表大小 K     = {original_vocab_size}")
    print(f"   CausalQwen词汇表大小    = {causal_vocab_size}")
    print(f"   预期CausalQwen大小 K+1  = {expected_causal_size}")
    print(f"   大小验证结果: {'✅ 正确 (K+1)' if size_correct else '❌ 错误'}")
    
    if size_correct:
        print(f"\n🎉 词汇表大小验证成功！")
        print(f"   理论: CausalQwen = Qwen + 1 (新增<NUM> token)")
        print(f"   实际: {causal_vocab_size} = {original_vocab_size} + 1")
    else:
        print(f"\n❌ 词汇表大小验证失败！")
        print(f"   实际差异: {causal_vocab_size - original_vocab_size}")
        
        # 分析可能的原因
        if causal_vocab_size == original_vocab_size:
            print(f"   可能原因: <NUM> token已存在于原始词汇表中")
        elif causal_vocab_size > expected_causal_size:
            print(f"   可能原因: 添加了多个token")
        else:
            print(f"   可能原因: token添加失败")
    
    return size_correct


def test_tokenization_functionality(causal_tokenizer):
    """测试分词功能"""
    print_separator("步骤6：分词功能验证", 2)
    
    test_texts = [
        "Hello world",  # 无数字
        "The price is 99.99 dollars",  # 单个数字
        "From 100 items, 25 were defective, costing 1250.50 total",  # 多个数字
        "No numbers here at all",  # 无数字
    ]
    
    print(f"测试文本分词和数值提取:")
    
    for i, text in enumerate(test_texts):
        print(f"\n文本 {i+1}: '{text}'")
        
        try:
            # 使用我们的tokenize_with_numbers方法
            if hasattr(causal_tokenizer, 'tokenize_with_numbers'):
                tokens, numerical_values = causal_tokenizer.tokenize_with_numbers(text)
                
                print(f"   tokens: {tokens}")
                print(f"   数值: {numerical_values}")
                
                # 检查<NUM> token的使用
                num_token_count = tokens.count('<NUM>')
                non_zero_values = sum(1 for val in numerical_values if val != 0.0)
                
                print(f"   <NUM> token数量: {num_token_count}")
                print(f"   非零数值数量: {non_zero_values}")
                print(f"   数量匹配: {'✅' if num_token_count == non_zero_values else '❌'}")
            else:
                print(f"   ⚠️ tokenize_with_numbers方法不存在")
                
        except Exception as e:
            print(f"   ❌ 分词失败: {e}")


def test_batch_encoding(causal_tokenizer):
    """测试批量编码功能"""
    print_separator("步骤7：批量编码验证", 2)
    
    test_texts = [
        "The item costs 99.99 dollars.",
        "From a batch of 100 items.",
        "Regular text without numbers."
    ]
    
    try:
        print(f"测试批量编码...")
        inputs = causal_tokenizer.batch_encode_plus(
            test_texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        
        print(f"✅ 批量编码成功")
        print(f"   input_ids形状: {inputs['input_ids'].shape}")
        print(f"   attention_mask形状: {inputs['attention_mask'].shape}")
        print(f"   numerical_values形状: {inputs['numerical_values'].shape}")
        
        # 检查是否有<NUM> token
        num_token_id = causal_tokenizer.num_token_id
        num_token_positions = (inputs['input_ids'] == num_token_id)
        num_token_count = num_token_positions.sum().item()
        
        print(f"   批次中<NUM> token数量: {num_token_count}")
        
        # 检查数值数组中的非零值
        non_zero_numerical = (inputs['numerical_values'] != 0.0).sum().item()
        print(f"   批次中非零数值数量: {non_zero_numerical}")
        print(f"   数量匹配: {'✅' if num_token_count == non_zero_numerical else '❌'}")
        
        return inputs
        
    except Exception as e:
        print(f"❌ 批量编码失败: {e}")
        return None


def comprehensive_verification_summary(
    original_vocab_size, causal_vocab_size, size_correct, causal_num_token_id,
    architecture_correct, weights_inherited
):
    """综合验证总结"""
    print_separator("最终验证总结", 1)
    
    print(f"📊 词汇表验证结果:")
    print(f"   原始Qwen词汇表大小:    K = {original_vocab_size}")
    print(f"   CausalQwen词汇表大小:      {causal_vocab_size}")
    print(f"   理论值 K+1:            K+1 = {original_vocab_size + 1}")
    print(f"   大小验证:              {'✅ 通过' if size_correct else '❌ 失败'}")
    
    print(f"\n🔢 <NUM> Token 验证结果:")
    print(f"   <NUM> token ID:        {causal_num_token_id}")
    print(f"   ID有效性:              {'✅ 有效' if causal_num_token_id >= 0 else '❌ 无效'}")
    
    print(f"\n🏗️ 模型架构验证结果:")
    print(f"   权重形状验证:          {'✅ 通过' if architecture_correct else '❌ 失败'}")
    print(f"   权重继承验证:          {'✅ 通过' if weights_inherited else '❌ 失败'}")
    
    # 最终判断
    all_passed = size_correct and causal_num_token_id >= 0 and architecture_correct and weights_inherited
    
    print(f"\n🏆 最终验证结果:")
    if all_passed:
        print(f"   ✅ 所有验证通过！")
        print(f"   ✅ CausalQwen词汇表 = Qwen词汇表 + 1")
        print(f"   ✅ <NUM> token正确添加且功能正常")
        print(f"   ✅ 模型权重形状符合预期 [K+1, d_model]")
        print(f"   ✅ 权重继承机制工作正常")
        print(f"\n🎯 结论: 词汇表设计完全符合理论要求 (K+1)，模型架构验证通过！")
    else:
        print(f"   ❌ 验证失败，存在以下问题:")
        if not size_correct:
            print(f"      - 词汇表大小不正确")
        if causal_num_token_id < 0:
            print(f"      - <NUM> token ID无效")
        if not architecture_correct:
            print(f"      - 模型权重形状不正确")
        if not weights_inherited:
            print(f"      - 权重继承失败")
        print(f"\n🔧 建议: 检查相关模块的实现")
    
    return all_passed


def main():
    """主函数"""
    print_separator("CausalQwen 完整验证脚本 (扩展版)", 1)
    print("目标：彻底验证 CausalQwen 相对于 Qwen 的完整设计")
    print("验证内容：词汇表、模型权重形状、权重继承、分词功能")
    
    # 配置
    model_path = "~/models/Qwen2.5-0.5B"
    
    print(f"\n配置信息:")
    print(f"   Qwen模型路径: {model_path}")
    print(f"   展开后路径: {os.path.expanduser(model_path)}")
    
    # 执行验证步骤
    
    # 步骤1：验证原始Qwen
    original_tokenizer, original_vocab_size, original_has_num = verify_original_qwen_tokenizer(model_path)
    if original_tokenizer is None:
        print("❌ 无法加载原始Qwen分词器，验证终止")
        return
    
    # 步骤2：验证CausalQwen
    causal_tokenizer, causal_vocab_size, causal_num_token_id = verify_causal_qwen_tokenizer(model_path)
    if causal_tokenizer is None:
        print("❌ 无法初始化CausalQwen分词器，验证终止")
        return
    
    # 步骤3：模型架构验证
    qwen_model, causal_model, architecture_correct = verify_model_architectures(model_path, causal_tokenizer)
    if qwen_model is None or causal_model is None:
        print("❌ 无法加载模型，验证终止")
        return
    
    # 步骤4：权重继承验证
    weights_inherited = verify_weight_inheritance(qwen_model, causal_model, causal_tokenizer)
    
    # 步骤5：词汇表对比验证
    size_correct = compare_vocabularies(
        original_tokenizer, original_vocab_size, 
        causal_tokenizer, causal_vocab_size
    )
    
    # 步骤6：功能测试
    test_tokenization_functionality(causal_tokenizer)
    
    # 步骤7：批量编码测试
    test_batch_encoding(causal_tokenizer)
    
    # 最终总结
    all_passed = comprehensive_verification_summary(
        original_vocab_size, causal_vocab_size, size_correct, causal_num_token_id,
        architecture_correct, weights_inherited
    )
    
    # 输出建议
    if not all_passed:
        print_separator("问题诊断与建议", 2)
        print("可能的问题和解决方案:")
        print("1. 检查QwenTokenizerWrapper.__init__中的token添加逻辑")
        print("2. 确认<NUM> token确实被添加到了词汇表中")
        print("3. 验证ActionNetwork权重形状和初始化逻辑")
        print("4. 检查知识传输机制的实现")


if __name__ == "__main__":
    main() 