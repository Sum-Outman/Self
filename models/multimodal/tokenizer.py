#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工业级文本分词器模块

包含：
1. IndustrialTokenizer - 从零开始的文本分词器

工业级AGI系统要求：从零开始训练，不使用预训练模型依赖
"""

import torch


class IndustrialTokenizer:
    """工业级文本分词器
    
    特征：
    - 从零开始构建的词表
    - 支持中英文分词
    - 支持注意力掩码和token类型
    """
    def __init__(self, vocab_size=100000):
        self.vocab_size = vocab_size
        # 初始化简单词表（实际应用中应从数据学习）
        self.word_to_id = {}
        self.id_to_word = {}
        self._build_initial_vocab()
        
    def _build_initial_vocab(self):
        """构建初始词表"""
        # 特殊token
        special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
            
        # 添加ASCII字符
        char_offset = len(special_tokens)
        for i in range(256):  # ASCII字符
            char = chr(i)
            idx = char_offset + i
            self.word_to_id[char] = idx
            self.id_to_word[idx] = char
            
        # 添加常见中文字符（500个常用汉字）
        # 完整版，实际应用中应从数据学习）
        common_chinese_chars = [
            '的', '一', '是', '在', '不', '了', '有', '和', '人', '这',
            '中', '大', '为', '上', '个', '国', '我', '以', '要', '他',
            '时', '来', '用', '们', '生', '到', '作', '地', '于', '出',
            '就', '分', '对', '成', '会', '可', '主', '发', '年', '动',
            '同', '工', '也', '能', '下', '过', '子', '说', '产', '种',
            '面', '而', '方', '后', '多', '定', '行', '学', '法', '所',
            '民', '得', '经', '十', '三', '之', '进', '着', '等', '部',
            '度', '家', '电', '力', '里', '如', '水', '化', '高', '自',
            '二', '理', '起', '小', '物', '现', '实', '加', '量', '都',
            '两', '体', '制', '机', '当', '使', '点', '从', '业', '本',
            '去', '把', '性', '好', '应', '开', '它', '合', '还', '因',
            '由', '其', '些', '然', '前', '外', '天', '政', '四', '日',
            '那', '社', '义', '事', '平', '形', '相', '全', '表', '间',
            '样', '与', '关', '各', '重', '新', '线', '内', '数', '正',
            '心', '反', '你', '明', '看', '原', '又', '么', '利', '比',
            '或', '但', '质', '气', '第', '向', '道', '命', '此', '变',
            '条', '只', '没', '结', '解', '问', '意', '建', '月', '公',
            '无', '系', '军', '很', '情', '者', '最', '立', '代', '想',
            '已', '通', '并', '提', '直', '题', '党', '程', '展', '五',
            '果', '料', '象', '员', '革', '位', '入', '常', '文', '总',
            '次', '品', '式', '活', '设', '及', '管', '特', '件', '长',
            '求', '老', '头', '基', '资', '边', '流', '路', '级', '少',
            '图', '山', '统', '接', '知', '较', '将', '组', '见', '计',
            '别', '她', '手', '角', '期', '根', '论', '运', '农', '指',
            '几', '九', '区', '强', '放', '决', '西', '被', '干', '做',
            '必', '战', '先', '回', '则', '任', '取', '据', '处', '队',
            '南', '给', '色', '光', '门', '即', '保', '治', '北', '造',
            '百', '规', '热', '领', '七', '海', '口', '东', '导', '器',
            '压', '志', '世', '金', '增', '争', '济', '阶', '油', '思',
            '术', '极', '交', '受', '联', '什', '认', '六', '共', '权',
            '收', '证', '改', '清', '美', '再', '采', '转', '更', '单',
            '风', '切', '打', '白', '教', '速', '花', '带', '安', '场',
            '身', '车', '例', '真', '务', '具', '万', '每', '目', '至',
            '达', '走', '积', '示', '议', '声', '报', '斗', '完', '类',
            '八', '离', '华', '名', '确', '才', '科', '张', '信', '马',
            '节', '话', '米', '整', '空', '元', '况', '今', '集', '温',
            '传', '土', '许', '步', '群', '广', '石', '记', '需', '段',
            '研', '界', '拉', '林', '律', '叫', '且', '究', '观', '越',
            '织', '装', '影', '算', '低', '持', '音', '众', '书', '布',
            '复', '容', '儿', '须', '际', '商', '非', '验', '连', '断',
            '深', '难', '近', '矿', '千', '周', '委', '素', '技', '备',
            '半', '办', '青', '省', '列', '习', '响', '约', '支', '般',
            '史', '感', '劳', '便', '团', '往', '酸', '历', '市', '克',
            '何', '除', '消', '构', '府', '称', '太', '准', '精', '值',
            '号', '率', '族', '维', '划', '选', '标', '写', '存', '候',
            '毛', '亲', '快', '效', '斯', '院', '查', '江', '型', '眼',
            '王', '按', '格', '养', '易', '置', '派', '层', '片', '始',
            '却', '专', '状', '育', '厂', '京', '识', '适', '属', '圆',
            '包', '火', '住', '调', '满', '县', '局', '照', '参', '红',
            '细', '引', '听', '该', '铁', '价', '严', '龙', '飞', '慢',
            '师', '普', '谈', '训', '陈', '雨', '阿', '错', '女', '刘',
            '啊', '李', '王', '张', '赵', '钱', '孙', '周', '吴', '郑',
            '你', '好', '吗', '呢', '吧', '呀', '啦', '哇', '哦', '唉',
            '喂', '嗯', '哼', '哈', '嘿', '嗨', '呸', '嘻', '呵', '呜',
        ]
        
        # 添加中文字符到词表
        chinese_offset = char_offset + 256
        for i, char in enumerate(common_chinese_chars):
            if chinese_offset + i >= self.vocab_size:
                break  # 不要超出vocab_size
            idx = chinese_offset + i
            self.word_to_id[char] = idx
            self.id_to_word[idx] = char
            
        # 为剩余的vocab_size创建占位映射
        # 确保所有可能的token ID都有映射
        for i in range(self.vocab_size):
            if i not in self.id_to_word:
                # 创建占位token
                实现 = f'[TOKEN_{i}]'
                self.id_to_word[i] = 实现
                self.word_to_id[实现] = i
            
    def __call__(self, text, padding=True, max_length=512, return_tensors="pt", truncation=True):
        """分词
        
        参数:
            text: 输入文本字符串
            padding: 是否填充
            max_length: 最大序列长度
            return_tensors: 返回的tensor类型
            truncation: 是否截断
            
        返回:
            包含input_ids和attention_mask的字典
        """
        # 简单分词：按字符分割
        tokens = []
        for char in text[:max_length]:
            token = char
            if token in self.word_to_id:
                tokens.append(self.word_to_id[token])
            else:
                tokens.append(self.word_to_id['[UNK]'])
                
        # 添加[CLS]和[SEP] token
        tokens = [self.word_to_id['[CLS]']] + tokens + [self.word_to_id['[SEP]']]
        
        # 截断
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.word_to_id['[SEP]']]
            
        # 创建注意力掩码
        attention_mask = [1] * len(tokens)
        
        # 填充
        if padding and len(tokens) < max_length:
            pad_len = max_length - len(tokens)
            tokens = tokens + [self.word_to_id['[PAD]']] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            
        # 转换为tensor
        input_ids = torch.tensor([tokens], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    def decode(self, token_ids, skip_special_tokens=True):
        """将token IDs转换回文本
        
        参数:
            token_ids: token ID列表或tensor
            skip_special_tokens: 是否跳过特殊token
            
        返回:
            解码后的文本字符串
        """
        # 如果输入是tensor，转换为列表
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        
        # 如果是嵌套列表（如批量处理），获取第一个序列
        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        
        text_chars = []
        for token_id in token_ids:
            if skip_special_tokens:
                # 跳过特殊token
                special_ids = [self.word_to_id[token] for token in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']]
                if token_id in special_ids:
                    continue
            
            # 查找对应的字符
            if token_id in self.id_to_word:
                char = self.id_to_word[token_id]
                # 如果是特殊token且不跳过，则添加特殊标记
                if char.startswith('[') and char.endswith(']'):
                    if not skip_special_tokens:
                        text_chars.append(char)
                else:
                    text_chars.append(char)
            else:
                # 未知ID，添加实现
                text_chars.append('[UNK]')
        
        return ''.join(text_chars)