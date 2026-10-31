"""调度的可见关系：生成位置 i 能看到 k，当且仅当 k 生成得更早且所在块不在 i 右边。

序列 = prompt（完全已知，内部普通因果）+ blocks 个 L 词生成块。调度：
  causal              普通从左到右（上界）
  delayed             第 t 块只看到 ≤t-2 块与本块已生成前缀（现有延迟反馈的条件）
  forward --stagger s 块内正序，相邻块错开 s 步开始
  reverse --stagger s 块内倒序（先生成第 L 个词），相邻块错开 s 步开始
  reverse --stagger s --segment g  按段倒序：块切成长 g 的段，段间倒序、段内正序
                                  （g=1 即逐词倒序）
  landing             PSF 式：所有块同时正序开始，第 2 个词起额外读右侧块首词
  meet --stagger d    用户的原始方案：两块一组同时解码，前一块正序、后一块倒序，
                      在交界处会合；后一块晚 d 步，所以它的块首能读到前一块的块尾
  alternate --stagger d  推广：所有块交替正序/倒序，每块比前一块晚 d 步
s=0 时所有块同时开始（完全并行）。除 landing 外，所有调度都不读右侧的块。生成位置 i 能看到 k，当且仅当 k 生成得更早且所在块
不在 i 右边；reverse 下丢掉的词对距离一律 ≥ L+s，见
docs/idea-reverse-staggered-decoding.md。
"""


import torch


SCHEDULES = ("causal", "delayed", "forward", "reverse", "landing", "meet", "alternate")


def visibility(schedule, prompt, blocks, block_size, stagger=None, segment=1):
    """返回 [N, N] 布尔矩阵 V：V[i, k] 表示生成位置 i 时可以读到位置 k 的词。"""
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    if prompt < 1 or blocks < 1 or block_size < 1:
        raise ValueError("prompt, blocks and block_size must be positive")
    n = prompt + blocks * block_size
    position = torch.arange(n)
    generated = position >= prompt
    block = torch.where(generated, (position - prompt) // block_size, -1)
    slot = torch.where(generated, (position - prompt) % block_size, 0)
    if schedule == "delayed":
        same = block[:, None] == block[None, :]
        allowed = (block[None, :] <= block[:, None] - 2) | (
            same & (slot[None, :] < slot[:, None])
        )
    else:
        if schedule == "causal":
            order, step = slot, block_size
        elif schedule == "landing":
            order, step = slot, 0
        else:
            if stagger is None or not 0 <= stagger <= block_size:
                raise ValueError("stagger must be in [0, block_size]")
            if schedule == "forward":
                order = slot
            elif schedule in ("meet", "alternate"):
                # 偶数块正序、奇数块倒序：相邻两块从各自外侧写向共同的交界处。
                order = torch.where(block % 2 == 0, slot, block_size - 1 - slot)
            else:
                if segment < 1 or block_size % segment:
                    raise ValueError("segment must divide block_size")
                # 段间倒序：最后一段最先写，块头所在的第一段最后写；段内仍从左到右。
                last = block_size // segment - 1
                order = (last - slot // segment) * segment + slot % segment
            step = stagger
        start = block * step
        if schedule == "meet":
            # 两块一组同时解码，组内倒序块晚 stagger 步；组与组依次进行。
            start = (block // 2) * (block_size + stagger) + (block % 2) * stagger
        clock = start + order
        allowed = (clock[None, :] < clock[:, None]) & (block[None, :] <= block[:, None])
        if schedule == "landing":
            # 着陆目标：右侧块首词最先生成，本块其余词都能读到它。
            allowed |= (
                (block[None, :] == block[:, None] + 1)
                & (slot[None, :] == 0)
                & (slot[:, None] >= 1)
            )
    rows, keys = generated[:, None], generated[None, :]
    # prompt 完全已知：生成位置都能读；prompt 内部普通因果，读不到任何生成位置。
    allowed = torch.where(rows & ~keys, True, allowed)
    allowed = torch.where(~rows, ~keys & (position[None, :] < position[:, None]), allowed)
    return allowed


def rounds(visible, prompt):
    """每个生成位置最早能采样的轮次：所依赖的生成位置都已在更早轮次完成。"""
    n = visible.shape[0]
    depth = torch.zeros(n, dtype=torch.long, device=visible.device)
    generated = torch.arange(n, device=visible.device) >= prompt
    needs = visible & generated[None, :]
    for _ in range(n):
        update = torch.where(needs, depth[None, :] + 1, 0).max(1).values
        update = torch.where(generated, update, 0)
        if torch.equal(update, depth):
            break
        depth = update
    return depth[prompt:]


PLAN_SCHEDULES = ("causal", "forward", "reverse", "meet")


def token_clock(schedule, prompt, blocks, block_size, stagger=None, segment=1):
    """生成词的生成时刻 [blocks·L] 与每块起步时刻 [blocks]；与 visibility 的定义一致。"""
    slot = torch.arange(blocks * block_size) % block_size
    block = torch.arange(blocks * block_size) // block_size
    index = torch.arange(blocks)
    if schedule == "causal":
        order, starts = slot, index * block_size
    elif schedule == "forward":
        order, starts = slot, index * stagger
    elif schedule == "meet":
        order = torch.where(block % 2 == 0, slot, block_size - 1 - slot)
        starts = (index // 2) * (block_size + stagger) + (index % 2) * stagger
    else:
        last = block_size // segment - 1
        order = (last - slot // segment) * segment + slot % segment
        starts = index * stagger
    return (starts[block] + order).double(), starts.double()


def plan_visibility(schedule, prompt, blocks, block_size, codes, stagger=None, segment=1):
    """返回 [N, N] 可见矩阵，N = prompt + blocks·L（词）+ blocks·codes（计划码）。

    第 t 块计划码的时刻在该块起步时刻减 1 与起步时刻之间（第 k 个码依次排开），
    规则与词相同：k 对 i 可见当且仅当 k 更早生成且所在块不在 i 右边。
    """
    if schedule not in PLAN_SCHEDULES:
        raise ValueError(f"schedule must be one of {PLAN_SCHEDULES}")
    n = prompt + blocks * block_size
    total = n + blocks * codes
    words = visibility(schedule, prompt, blocks, block_size, stagger, segment)
    clock, starts = token_clock(schedule, prompt, blocks, block_size, stagger, segment)
    plan_block = torch.arange(blocks).repeat_interleave(codes)
    plan_index = torch.arange(blocks * codes) % codes
    plan_clock = starts[plan_block] - 1 + (plan_index + 1).double() / (codes + 1)
    word_block = torch.arange(blocks * block_size) // block_size
    time = torch.cat((clock, plan_clock))
    block = torch.cat((word_block, plan_block))
    generated = (time[None, :] < time[:, None]) & (block[None, :] <= block[:, None])
    allowed = torch.zeros(total, total, dtype=torch.bool)
    allowed[:n, :n] = words
    allowed[prompt:, prompt:] = generated
    # 词与词之间沿用 visibility（同一规则，保证一致）；计划行读取全部 prompt。
    allowed[prompt:n, prompt:n] = words[prompt:, prompt:]
    allowed[n:, :prompt] = True
    return allowed
