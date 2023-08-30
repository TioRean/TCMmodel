from itertools import combinations, product, chain, cycle, repeat, compress
from math import ceil, factorial as ft
import numpy as np
import copy
import re
import pandas as pd
from random import shuffle
# ------------------------规则生成医案数据模型------------------------

def accessory_only_del(content, optional, accessory):
    # 为了将accessory中子元素与父元素在以后的组合中配对，即子元素存在的情况下，父元素一定存在，
    # accessory_only_del将不符合上述条件的instance删除
    # content 通过optional_combination方法后得到的optional的所有组合病例
    # optional：初始的optional列表
    result = []
    used_accessory = {}
    for element in optional:
        if element in accessory:
            used_accessory[element] = accessory.get(element)
    for instance in content:
        instance_to_list = list(instance)
        skip = False
        for key in used_accessory:
            if key in instance:
                value = used_accessory.get(key)
                if value not in instance_to_list:
                    skip = True
        if not skip:
            result.append(instance_to_list)
    return result


def get_selector(target, total):
    # get_selector用于获得一个选择器，其结果是itertools.compress的selectors参数
    # 返回结果格式为[1, 1, 0, 0, 1, 0, 1, 1...]，为打乱的含有0或1的列表
    # 其中1表示获取某iterable的数据，0表示跳过该iterable的数据
    # target 是想在该iterable中获得取值的总个数，等于1的个数
    # total 是iterable的总个数，total - target等于0的个数
    arr = []
    arr.extend(np.ones(int(target)))
    arr.extend(np.zeros(int(total-target)))
    state = np.random.get_state()
    np.random.shuffle(arr)
    np.random.set_state(state)
    return arr


def combination(total, each):
    # 组合公式 即C(n,m) = n! / m!(n-m)! n为总个数，m为每次取的个数
    # 本方法中total就是n，each就是m
    return ft(total) / (ft(each) * ft(total - each))


def inner_optional_combination(optional, accessory, combi_max_len=8, optional_max=24,
                               BD_boundary=15, BD_len_boundary=5, BD_basic_combi_num=22818, BD_grow_step=5000):
    # inner_optional_combination用于对optional内部元素进行组合（排列组合的“组合”），每组optional可能取一个或多个，将所有可能组合后返回
    # 举例：[’发热‘,’汗出‘,’恶寒‘] -> [['发热'],[’汗出‘],[’恶寒‘],['发热',’汗出‘],['发热',’恶寒‘],[’汗出‘,’恶寒‘],['发热',’汗出‘,’恶寒‘]]

    # optional是备选元素库，通过不同的组合模拟现实中多变的临床
    # accessory 是一个字典，key为子元素，value为父元素, 子元素存在的情况下，父元素必定存在，否则该病例将被删除

    # combi_max_len是允许的组合公式C(n,m) = n! / m!(n-m)!中 m的最大值。(理论上m的取值为(1，n]，但是这样容易导致返回结果过多，即数据爆炸)
    # 而实际临床中，患者表现的临床症状是有限的，一个病机下表现出的症状更是有限，以3-6个症状为多，m值没必要取到n，故限定了combi_max_len参数
    # optional_max 是指optional列表的最大长度，设置本参数主要目的是防止数据爆炸
    # BD是big data的缩写，在组合过程中，返回结果的长度是以指数级增长，若不加限制，返回结果可轻易达到十万条以上，这里称这种情况为big data，这种结果对训练并没有好处
    # BD_boundary 若optional长度超过BD_boundary则称为big data，big data自身有特殊的组合方式，以防止数据爆炸
    # BD_len_boundary 在一个满足big data的optional中，当m取值<=BD_len_boundary时，用普通的组合方法，一旦超过此值，用big data有特殊的组合方式
    # BD_basic_combi_num big data返回的列表长度大于本参数，22818是当optional的len为15，combi_max_len为8时的返回列表长度
    # BD_grow_step 对big data人为限定返回结果的增长速度，从BD_boundary值（即15）开始，optional长度每增加1，返回结果长度增加BD_grow_step，防止数据爆炸

    result = []
    assert len(optional) <= optional_max
    if len(optional) <= BD_boundary:
        # 当optional长度小于等于BD_boundary，即非big data时的组合方式，结果为C(n,1)+C(n,2)...对m不同取值得组合结果都放置在result中
        # 遍历过程中，若m取值大于combi_max_len时停止
        for length in range(0, len(optional)):
            if length + 1 > combi_max_len:
                break
            else:
                result.extend(combinations(optional, length+1))
    elif len(optional) > BD_boundary and len(optional) <= optional_max:
        # 当optional长度大于BD_len_boundary，即big data时的组合方式
        assert combi_max_len > BD_len_boundary
        for length in range(1, BD_len_boundary):
            # 在（1，BD_len_boundary）范围内为普通的组合方式
            result.extend(combinations(optional, length))
        remain = BD_basic_combi_num + BD_grow_step * (len(optional) - BD_boundary) - len(result)
        for i in range(BD_len_boundary, combi_max_len + 1):
            # 在（BD_len_boundary, combi_max_len + 1）范围内，人为限定返回结果的增长速度
            # num_each_grade为每次组合时规定的产生组合的总个数，通过itertools方法进行随机筛选，产生的组合结果大于num_each_grade的部分删除
            # 若产生的组合结果小于num_each_grade，则取所有组合结果，本轮num_each_grade多余的部分通过留给下一轮
            num_each_grade = int(remain / (combi_max_len - i + 1))
            combi = combinations(optional, i)
            combi_num = combination(len(optional), i)
            if num_each_grade >= combi_num:
                result.extend(combi)
                remain -= combi_num
            else:
                selector = get_selector(target=num_each_grade, total=combination(len(optional), i))
                target = compress(data=combi, selectors=selector)
                result.extend(target)
                remain -= num_each_grade
    result = accessory_only_del(content=result, optional=optional, accessory=accessory)
    shuffle(result)
    return result


def get_num_of_combi_to_pdt(opt_combi_list, pdt_max_len):
    # get_num_of_combi_to_pdt 用于寻找使用combinations和product计算后结果长度不超过max_len的最大参与计算的列表个数
    # opt_combi_list 代表所有可能参与计算的列表，即[optional_N1,optional_N2, optional_N3...]的各自组合结果
    # opt_combi_list 举例：[[['发热'], ['发热', '往来寒热']], [['身体疼痛'], ['头痛'], ['身体疼痛', '头痛']]]
    count = 1
    for i in range(0, len(opt_combi_list)):
        count *= len(opt_combi_list[i])
        if count > pdt_max_len:
            return i if i != 0 else 0
    return len(opt_combi_list)


def optional_to_combi_list(optionals, accessory):
    # optional_to_combi_list 输入二维列表optionals，每个子列表optional取1个或多个值，将多个子列表所取的值依次进行组合产生新的列表，将产生的多个列表返回
    # 返回值是一个三维列表，举例：[[['发热'], ['发热', '往来寒热']], [['身体疼痛'], ['头痛'], ['身体疼痛', '头痛']]]
    # optionals为2维列表，内包含多个optional子列表,举例：[['发热', '往来寒热'], ['身体疼痛', '头痛']]
    # accessory参考上文
    result = []
    for i in range(0, len(optionals)):
        result.append(inner_optional_combination(optional=optionals[i], accessory=accessory))
    return result


def optional_combi_to_pdt(idx_product, optional_combi):
    # optional_combi_to_pdt 将optional_to_combi_list方法后（即optional组合后）产生的多个列表进行product（笛卡尔积）
    # product即每组大列表中取1个子列表进行拼接，产生不同的instance
    # idx_product 是用于product的大列表的索引，为了防止product之后产生过多的结果，通过本参数进行限制增长
    result = []
    if idx_product == 0:
        return optional_combi[0]
    elif idx_product >= 1:
        content = product(*optional_combi[0:idx_product])
        for instance_raw in content:
            # itertools.chain对列表进行降维
            # 将instance_raw中的null无用字符删除
            instance_cleaned = list(filter(lambda x: x != 'null', list(chain.from_iterable(instance_raw))))
            result.append(instance_cleaned)
        return result


def max_len_idx(arr):
    # 在一个多维列表中，找到子列表长度最大的列表对应的索引
    max_len = -1
    max_len_idx = -1
    for i in range(0, len(arr)):
        if len(arr[i]) > max_len:
            max_len = len(arr[i])
            max_len_idx = i
    return max_len_idx


def optionals_zipping(content):
    # 对content内部的子列表进行zip方法的拼接
    # 较短的子元素会用repeat方法进行扩容
    max_idx = max_len_idx(content)
    for i in range(0, len(content)):
        if i == max_idx:
            continue
        else:
            times = ceil(len(content[max_idx]) / len(content[i]))
            content[i] = list(chain.from_iterable(repeat(content[i], times)))
    result = list(zip(*content))
    return result


def multiple_optional_combination(
        accessory, optional_A=None, optional_B=None, optional_C=None,
        optional_D=None, optional_E=None, optional_F=None,
        optional_G=None, optional_H=None, product_max_num=67818):
    # 复杂中医模型往往涉及寒热错杂，虚实夹杂等复杂病机，这类病机很难用单个compulsory及单个optional列表进行组合
    # 举例来说，对于柴桂姜汤而言，寒、热、津亏、中阳亏虚都是重要的病机，若用单个compulsory及optional，则可能出现仅有单一病机的拼接结果
    # multiple_optional_combination 用于产生涉及复杂中医模型的optional所有组合
    # 该方法中有最多8个独立的列表（即备选元素库），分别代表该方证的8个维度
    # 将从每个optional列表中抽取一个或多个元素组合成新的列表后，再将所有的新列表在表与表之间进行组合
    # optional_x(x=A,B,C,D,E,F,G,H)代表备选元素库
    # product_max_num为product后返回列表所允许的最大长度，即产生医案的最大例数，该参数参与防止数据爆炸的机制
    result = []
    content = []
    # 在多组optional中每组抽出一个或多个元素进行拼接组成一个列表，用以表示一个方证（不完整，还需进一步拼接），每个方证称为一个instance
    optional_list = list(
        filter(lambda x: x is not None and x != [], [optional_A, optional_B, optional_C, optional_D, optional_E, optional_F,
                                                     optional_G, optional_H]))
    if len(optional_list) > 0:
        # 进行optional列表内部进行combination组合
        opt_combi_list = optional_to_combi_list(optionals=optional_list, accessory=accessory)
        while len(optional_list) > 0:
            # optionals列表中对combination后的多个列表进行product组合（笛卡尔积）
            # __get_num_of_combi_to_pdt方法用于控制product产生的长度，保证product返回列表不超过product_max_len
            num_to_product = get_num_of_combi_to_pdt(opt_combi_list, pdt_max_len=product_max_num)
            content.append(optional_combi_to_pdt(idx_product=num_to_product, optional_combi=opt_combi_list))
            del optional_list[:num_to_product]
            del opt_combi_list[:num_to_product]
        if len(content) > 1:
            content = optionals_zipping(content)
            for instance in content:
                result.append(list(chain.from_iterable(instance)))
        elif len(content) == 1:
            result = list(chain.from_iterable(content))
        return result
    else: return None


def tongue_pulse_appearance(
        t_nature=['null'], t_coating_thickness=['null'], t_coating_color=['null'],
        t_coating_humidity=['null'],t_coating_character=['null'],
        p_rate=['null'], p_rhythm=['null'], p_position=['null'], p_body=['null'],
        p_strength=['null'], p_fluency=['null'], p_tension=['null'], p_complex=['null']):
    # tongue_pulse_appearance 用于不同舌象、脉象的组合
    # 若某个维度可能取空，也可能有值，可将list设置为['脉浮','null']
    # 包含舌质、苔厚度、苔色、苔湿润度、苔质（共5维度）、脉率、脉律、脉位、脉体、脉力、脉流利度、脉紧张度（共8维度）合计13种维度
    # 由于舌形使用不多，且大多没有互斥性，故放在备选元素库中组合
    # 每种维度只有一个值，以确保该舌脉不出现前后矛盾的情况
    # t_appearance、t_coating_thickness、t_coating_color、t_coating_humidity、t_coating_character分别代表舌质、苔厚度、苔色、苔湿润度、苔质
    # p_rate、p_rhythm、p_position、p_body、p_strength、p_shape 分别代表脉率、脉律、脉位、脉体、脉力、脉势
    # 上述11种参数(维度)分别都是一个list，每个list下包含该维度下不同的值
    result = []
    tp_product = product(
        t_nature, t_coating_thickness, t_coating_color, t_coating_humidity, t_coating_character,
        p_rate, p_rhythm, p_position, p_body, p_strength, p_fluency, p_tension, p_complex)
    for instance_raw in tp_product:
        # 清除instance_raw中的无效字符'null'
        instance_cleaned = list(filter(lambda x: x != 'null', list(instance_raw)))
        result.append(instance_cleaned)
    return result


def overlap(long, short, exchange=False):
    # 将optional 和 compulsory及舌脉一一对应组合在一起，不产生额外长度的数据
    # 为了防止数据爆炸，不用product，使用本方法组合
    # long为较长的列表， short为较短的列表
    count = 0
    result = []
    for x in cycle(short):
        if count < len(long):
            if len(x) == 0:
                continue
            else:
                if exchange:
                    result.append([*x, *long[count]])
                else:
                    result.append([*long[count], *x])
        else:
            break
        count += 1
    return result


def montage(compulsory=None, optional=None, tongue_pulse=None, product_max_len=200):
    # montage 用于将compulsory、optional、tongue_pulse列表中的不同元素组合结果进行拼接
    # 实现将一个中医方证规则转化为多个病例，转化成的病例用于进行transformer训练
    # 1. compulsory 是核心元素，不同的病例中必然存在这些元素
    # 2. optional 是备选元素，可有可无，通过不同的组合模拟现实中多变的临床
    # 3. tongue_pulse 内含多种组合之后的舌脉信息，将与compulsory及optional拼接
    # product_max_len 若optional及tongue_pulse皆较小时，即二者相乘的结果小于product_max_len时，用product方法适当增加返回结果的数量，防止返回结果过于疏散
    content = []
    result = []
    if compulsory is not None:
        if tongue_pulse is not None:
            content = copy.copy(tongue_pulse)
            for i in range(len(compulsory)):
                for instance in content:
                    instance.insert(i, compulsory[i])
        else:
            content = copy.copy([compulsory])
    elif compulsory is None and tongue_pulse is not None:
        content = copy.copy(tongue_pulse)
    elif compulsory is None and tongue_pulse is None:
        if optional is not None:
            return optional
    if optional is not None:
        if len(optional) * len(content) > product_max_len:
            if len(optional) >= len(content):
                result = overlap(long=optional, short=content)
            else:
                result = overlap(long=content, short=optional, exchange=True)
        else:
            instances_raw = product(optional, content)
            for instance_raw in instances_raw:
                # 将instance从instance_raw二维列表转化为instance_flatten一维列表
                instance_flatten = list(chain.from_iterable(instance_raw))
                result.append(instance_flatten)
    else:
        result = content
    return result


#------------------规则生成医案数据模型的使用工具及数据处理工具---------------------

class Instances():
    # 规则生成医案数据模型的类
    def __init__(self,compulsory, optional, tongue_pulse, medicine):
        accessory = {'汗出不彻': '发汗后', '发汗过多': '发汗后', '寒热如疟': '发热', '往来寒热': '发热', '潮热': '发热',
                 '反复发热': '发热','背恶寒': '恶寒', '大汗出': '汗出', '头汗出': '汗出', '手足汗出': '汗出', '自汗': '汗出',
                 '四肢厥冷': '手足冷','关节疼痛': '肢体疼痛', '肌肉酸痛': '肢体疼痛', '肢体游走性疼痛': '肢体疼痛', '起则头眩': '头晕',
                 '食即头眩': '头晕','口干但欲漱水不欲咽': '口干', '消渴': '口渴', '喘憋': '胸胁满闷', '哮喘': '胸胁满闷', '痰白': '咳痰',
                 '痰黄': '咳痰','不能食': '纳少', '欲吐不吐': '呕吐', '朝食暮吐': '呕吐', '腹中急痛': '腹痛', '绕脐痛': '腹痛',
                 '时腹自痛': '腹痛','腹满痛': '腹痛', '自利清水': '腹泻', '完谷不化': '腹泻', '里急后重': '腹泻',
                 '大便＞3日未解': '大便干结','大便先硬后溏': '大便干结', '嗜睡': '意识障碍', '昏迷': '意识障碍', '烦躁': '意识障碍',
                 '谵语': '意识障碍','发狂': '意识障碍', '反射性晕厥': '晕厥', '身黄如橘子色': '黄疸'}

        self.compulsory = compulsory

        optional_N1 = optional[0]
        optional_N2 = optional[1]
        optional_N3 = optional[2]
        optional_N4 = optional[3]
        optional_N5 = optional[4]
        optional_N6 = optional[5]
        optional_N7 = optional[6]
        optional_N8 = optional[7]
        self.optional = \
            multiple_optional_combination(accessory=accessory, optional_A=optional_N1, optional_B=optional_N2,
                                          optional_C=optional_N3, optional_D=optional_N4, optional_E=optional_N5,
                                          optional_F=optional_N6, optional_G=optional_N7, optional_H=optional_N8)

        self.tongue_pulse = \
            tongue_pulse_appearance(
                t_nature=tongue_pulse[0], t_coating_thickness=tongue_pulse[1], t_coating_color=tongue_pulse[2],
                t_coating_humidity=tongue_pulse[3], t_coating_character=tongue_pulse[4],
                p_rate=tongue_pulse[5], p_rhythm=tongue_pulse[6], p_position=tongue_pulse[7],
                p_body=tongue_pulse[8], p_strength=tongue_pulse[9], p_fluency=tongue_pulse[10],
                p_tension=tongue_pulse[11], p_complex=tongue_pulse[12]
            )

        self.medicine = medicine

    def get_source(self):
        # 通过规则生成模型的实例化对象，生成source，即包含必要元素、备选元素、舌脉的医案
        return montage(compulsory=self.compulsory, optional=self.optional, tongue_pulse=self.tongue_pulse)

    def get_target(self):
        # 获取对应的方药
        return self.medicine


def get_elements(dataframe_loc):
    # get_elements方法用于传入一个dataframe_loc对应的值或者series，返回相应的元素
    if type(dataframe_loc) == str:
        if dataframe_loc == 'None':
            return []
        else:
            return [dataframe_loc.replace('，',', ')]
    else:
        return list(filter(lambda x: x != 'None', dataframe_loc))


def tongue_pulse_add(arr):
    if arr == []:
        return ['null']
    else: return arr

def get_elements_list_for_tongue_pulse(dataframe, i):
    # 专门用于获取某组数据舌脉的方法
    content = []
    t_nature = get_elements(dataframe.loc[i, '舌淡白':'舌青'])
    content.append(tongue_pulse_add(t_nature))

    t_coating_color = get_elements(dataframe.loc[i, '白苔':'黑苔'])
    content.append(tongue_pulse_add(t_coating_color))
    t_coating_thickness = get_elements(dataframe.loc[i, '苔少':'苔厚'])
    content.append(tongue_pulse_add(t_coating_thickness))
    t_coating_humidity = get_elements(dataframe.loc[i, '苔水滑':'null'])
    content.append(tongue_pulse_add(t_coating_humidity))
    t_coating_character = get_elements(dataframe.loc[i, '苔腻':'null.1'])
    content.append(tongue_pulse_add(t_coating_character))

    p_rate = get_elements(dataframe.loc[i, '脉数':'null.2'])
    content.append(tongue_pulse_add(p_rate))
    p_rhythm = get_elements(dataframe.loc[i, '脉促':'null.3'])
    content.append(tongue_pulse_add(p_rhythm))
    p_position = get_elements(dataframe.loc[i, '脉浮':'null.4'])
    content.append(tongue_pulse_add(p_position))
    p_body = get_elements(dataframe.loc[i, '脉大':'null.5'])
    content.append(tongue_pulse_add(p_body))
    p_strength = get_elements(dataframe.loc[i, '脉虚':'null.6'])
    content.append(tongue_pulse_add(p_strength))
    p_fluency = get_elements(dataframe.loc[i, '脉滑':'null.7'])
    content.append(tongue_pulse_add(p_fluency))
    p_tension = get_elements(dataframe.loc[i, '脉弦':'null.8'])
    content.append(tongue_pulse_add(p_tension))
    p_complex = get_elements(dataframe.loc[i, '革脉':'null.9'])
    content.append(tongue_pulse_add(p_complex))
    return content


def save_item(instances_content, save_num, path):
    with open(path, 'a') as f:
        for index in range(0, save_num):
            f.write(instances_content[index])
            f.flush()
        f.close()
    instances_content = instances_content[save_num:]
    return instances_content

def save(instances_content, train_path='train_data.txt', val_path='val_data.txt', test_path='test_data.txt',
         val_ratio=0.10, test_ratio=0.05):
    assert val_ratio < 1 and val_ratio >= 0 and test_ratio < 1 and test_ratio >= 0 and val_ratio + test_ratio < 1
    train_ratio = 1 - val_ratio - test_ratio
    # 获取每个数据集的数目
    instances_num = len(instances_content)
    test_save_num = int(instances_num * test_ratio)
    val_save_num = int(instances_num * val_ratio)
    train_save_num = int(instances_num * train_ratio)
    print(f'训练集数: {train_save_num}')
    print(f'验证集数: {val_save_num}')
    print(f'测试集数: {test_save_num}')
    print(f'生成医案总数: {train_save_num + val_save_num + test_save_num}')

    # 保存test数据集
    if test_ratio > 0:
        instances_content = save_item(instances_content,test_save_num, test_path)
    # 保存val数据集
    if val_ratio > 0:
        instances_content = save_item(instances_content, val_save_num, val_path)
    # 保存train数据集
    save_item(instances_content, train_save_num, train_path)

#----------------------数据处理区-----------------------------

path = './data.xlsx'
df = pd.read_excel(path)
df = df.fillna('None')
# df_start df_end与df_mid的数据结构不同，分开处理
df_start = df.loc[:, :'备选元素库H'].astype(str)
df_mid = df.loc[:, :'null.9'].astype(str)
df_end = df.loc[:, :'药物20'].astype(str)

# 将df_end中的值'1'(原表格中表示存在该元素)，改为表头对应的值
columns = list(df_mid.columns)
for column in columns:
    if re.match(r'null\.\d+', column):
        temp = 'null'
        df_mid.loc[:, column] = df_mid[column].str.replace('1.0', temp, regex=False)
        df_mid.loc[:, column] = df_mid[column].str.replace('1', temp, regex=False)
    else:
        df_mid.loc[:, column] = df_mid[column].str.replace('1.0', column, regex=False)
        df_mid.loc[:, column] = df_mid[column].str.replace('1', column, regex=False)

# 获取每个instance中的相应元素，包括compulsory， optional， tongue， pulse， medicine
content = []
for i in range(0,len(df)):

    # 获取compulsory的元素
    compulsory = get_elements(df.loc[i,'必要元素库'])

    # 获取optional的元素
    optionals = get_elements(df_start.loc[i, '备选元素库A':'备选元素库H'])
    optionals = [optional.split('，') for optional in optionals]
    while len(optionals) < 8:
        optionals.append([])

    # 获取舌脉的元素
    tongue_pulse = get_elements_list_for_tongue_pulse(dataframe=df_mid, i=i)

    # 获取药物的元素
    medicines = get_elements(df_end.loc[i, '药物1':'药物20'])

    # 通过实例化instances类，获取数据的source（即方证，舌脉等）、target（即药物）
    inst = Instances(compulsory, optionals, tongue_pulse, medicines)
    sources = inst.get_source()
    target = inst.get_target()

    # 将source及target组合成一个病例（instance）
    # source与target用制表符（\t）隔开，末尾加空格符（\n）
    # 将一个line中的无用符号删除，包括[]'
    for source in sources:
        line = str(source) + '\t' + str(target) +'\n'
        line = line.replace('[','').replace(']','').replace("'",'')
        content.append(line)

# 将获得的所有instance打散
shuffle(content)

# 数据存储
save(content, train_path='train_small.txt', val_path='val_small.txt', test_path='test_small.txt', val_ratio=0, test_ratio=0)

