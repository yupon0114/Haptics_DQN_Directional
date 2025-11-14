# pico側 (MicroPython)
import rp2
from machine import Pin
import time
import sys

# --- PIO: 入力立ち上がり検出用 (元のアセンブリをほぼ維持) ---
@rp2.asm_pio(set_init=rp2.PIO.IN_LOW)
def edge_capture():
    label("start")
    pull(block)                 # OSR から X に初期値を入れる
    mov(x, osr)
    set(pindirs, 0b000)
    wait(0, pin, 0)
    wait(1, pin, 0)
    label("wait_for_fall")
    nop() [31]
    jmp(pin, "dec_and_fall")
    jmp("wait_for_rise")
    label("dec_and_fall")
    jmp(x_dec, "wait_for_fall")
    jmp("done")
    label("wait_for_rise")
    jmp(pin, "got_rise")
    jmp(x_dec, "wait_for_rise")
    jmp("done")
    label("got_rise")
    mov(isr, x)
    push()
    jmp("start")
    label("done")
    jmp("start")

# --- ユーティリティ関数 ---
def filter_large_values(data):
    """(最大値 + 最小値) / 2 より小さい値を除外"""
    if not data:
        return []
    max_val = max(data)
    min_val = min(data)
    threshold = (max_val + min_val) / 2
    return [x for x in data if x >= threshold]

def calculate_statistics(data):
    """平均・標準偏差を計算（データ長0/1に対応）"""
    n = len(data)
    if n == 0:
        return 0.0, 0.0, []
    mean = sum(data) / n
    if n == 1:
        stdev = 0.0
    else:
        # 標本分散（n-1 で割る）
        var = sum((x - mean) ** 2 for x in data) / (n - 1)
        stdev = var ** 0.5
    return mean, stdev, data

# --- GPIO設定 ---
trigger_pin = Pin(12, Pin.IN, pull=Pin.PULL_DOWN)  # トリガー用（in_baseに使う）
pins_capture = [13]     # キャプチャ対象ピンリスト

# --- ステートマシン準備 ---
sms = []
for i, pin_num in enumerate(pins_capture):
    cap_pin = Pin(pin_num, Pin.IN, pull=Pin.PULL_UP)
    # freq と in_base/jmp_pin は環境に合わせて調整してください
    sm = rp2.StateMachine(i, edge_capture, freq=20_000_000, in_base=trigger_pin, jmp_pin=cap_pin)
    sms.append(sm)

print("PICO ready. Waiting for commands on USB serial (READ).")

# --- メインループ: シリアルからコマンドを受け取る ---
while True:
    # USB シリアルからの行をブロッキングで待つ
    try:
        cmd = sys.stdin.readline()
    except Exception as e:
        # 環境によっては KeyboardInterrupt などが来ることもある
        print("stdin read error:", e)
        cmd = ""
    if not cmd:
        # 空行（タイムアウト等）ならループ継続
        continue
    cmd = cmd.strip()
    if cmd == "READ":
        print("READ")
        # FIFO のクリア（state machine の FIFO を空にする）
        for sm in sms:
            try:
                # rx_fifo() may not exist on all builds; try to drain safely
                while sm.rx_fifo():
                    _ = sm.get()
            except Exception:
                # もし rx_fifo() が無ければ、try-get を軽く回す
                try:
                    while True:
                        _ = sm.get()
                except Exception:
                    pass

        print("Starting trigger SM...")
        for sm in sms:
            sm.active(1)

        # キャプチャデータ収集（最大20回試行）
        results = {pin: [] for pin in pins_capture}
        attempts = 20
        for _ in range(attempts):
            # sm.put によってカウントを与えて PIO を動作させる
            for sm in sms:
                try:
                    sm.put(20000)
                except Exception:
                    pass
            
            # 少し待ってから結果を取りに行く（PIO のタイミングに合わせる）
            time.sleep(0.001)

            for pin, sm in zip(pins_capture, sms):
                try:
                    cycle_get = sm.get()  # 取得。失敗したら例外処理へ
                except Exception:
                    # 取得できなければスキップ
                    continue

                # 元のコードのロジックに合わせ、閾値で無視する
                if cycle_get > 19000:
                    # 無効値（タイムアウト等）
                    pass
                else:
                    clock = 20000 - cycle_get
                    clockCycles = clock * 2
                    # 20 MHz => 50 ns per cycle, 50ns * clockCycles -> ns
                    time_ns = clockCycles * 50  # ns
                    time_us = time_ns / 1000.0  # μs
                    results[pin].append(time_us)
        # 表示（取得できた分だけ）
        max_len = max((len(v) for v in results.values()), default=0)
        for i in range(max_len):
            line_parts = []
            for pin in pins_capture:
                vals = results[pin]
                if i < len(vals):
                    line_parts.append(f"GPIO{pin} = {vals[i]:.1f} μs")
                else:
                    line_parts.append(f"GPIO{pin} = -")
            print(f"{i+1} 回目: " + ", ".join(line_parts))
        print("Capture finished.")
        # --- 統計処理 ---
        for pin, data in results.items():
            normal_stats = calculate_statistics(data)
            filtered_data = filter_large_values(data)
            filtered_stats = calculate_statistics(filtered_data)

            # 元データ統計
            print(f"GPIO{pin} (元データ): 平均 = {normal_stats[0]:.1f} μs, 標準偏差 = {normal_stats[1]:.3f} μs, count = {len(data)}")
            # フィルタ後統計
            print(f"GPIO{pin} (フィルタ後): 平均 = {filtered_stats[0]:.1f} μs, 標準偏差 = {filtered_stats[1]:.3f} μs, データ = {filtered_stats[2]}")
    elif cmd == "RESET":
        print("RESET")
        machine.reset()
