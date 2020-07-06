import configparser

def get_config(config_file = 'config.ini'):
    parser = configparser.ConfigParser()
    parser.read(config_file)

    # 获取参数，以key-value的形式保存
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]

    # 元组数组直接转换为字典格式
    return dict(_conf_floats + _conf_ints + _conf_strings)


if __name__ == '__main__':
    res = get_config()
    print(res)