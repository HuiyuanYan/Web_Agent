from itertools import chain

SPECIAL_KEYS = ["@", "#", "$"]
ASCII_CHARSET = ["a", "b", "c"]
FREQ_UNICODE_CHARSET = ["€", "¥", "£"]

_key2id: dict[str, int] = {
    key: i
    for i, key in enumerate(
        chain(SPECIAL_KEYS, ASCII_CHARSET, FREQ_UNICODE_CHARSET, ["\n"])
    )
}
_id2key: list[str] = sorted(_key2id, key=_key2id.get)

# 生成所有中文字符并添加到整数列表
chinese_charset = [codepoint for codepoint in range(0x4E00, 0x9FFF + 1)]

for i in range(0,len(chinese_charset)):
    print(chr(chinese_charset[i]),end='')
