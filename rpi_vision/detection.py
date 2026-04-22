import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import serial

# ===================== CONFIG =====================
CAM_DEVICE = 0
MODEL_PATH = "models/gating_model_float16.tflite"

CAP_W, CAP_H = 640, 480
FORCE_MJPG = True

THREADS = 4
PROCESS_EVERY = 1  # CNN gating cada N frames (1 = siempre)

# ---- Gate (CNN SIEMPRE) ----
TH_ON  = 0.80
TH_OFF = 0.60
ALPHA_GATE  = 0.20
HOLD_FRAMES = 15

# ---- HSV red mask ----
S_MIN = 100
V_MIN = 60
H1_LOW, H1_HIGH = 0, 8
H2_LOW, H2_HIGH = 172, 180
KERNEL = 5
ERODE_IT = 1
DILATE_IT = 2
MIN_AREA = 700
MAX_AREA_FRAC = 0.40
MIN_RED_RATIO = 0.18

# ---- UART ----
SERIAL_PORT = "/dev/serial0"
BAUD = 115200
SEND_HZ = 30  # más alto ayuda a que los pulsos sean finos
send_dt = 1.0 / float(SEND_HZ)

# ---- PAN (360) por PULSOS ----
PAN_SIGN = -1 
PAN_PULSE_CMD = 45

PAN_DEAD_PIX = 28     # zona muerta en pixeles alrededor del crosshair 
PAN_ERR_FULL_PIX = 220 

PAN_ON_MIN_MS = 14    
PAN_ON_MAX_MS = 55     
PAN_OFF_MS    = 38    
PAN_DIR_COOLDOWN_MS = 120 


STOP_WHEN_LOST = True
LOST_STOP_SEC = 0.25   

# ---- TILT (igual que tuyo, funciona bien) ----
TILT_MIN = 1100
TILT_MAX = 1430
TILT_CENTER = (TILT_MIN + TILT_MAX) // 2

KP_TILT_US = 160.0
TILT_DEADZONE = 0.04
TILT_SMOOTH = 0.22
TILT_INVERT = True

# ---- CPU/cam ----
MAX_PROC_FPS = 25.0
# ================================================

def clamp(x, a, b):
    return a if x < a else b if x > b else x

# ---------- TFLite ----------
def make_interpreter(path, threads=4):
    itp = tflite.Interpreter(model_path=path, num_threads=threads)
    itp.allocate_tensors()
    return itp

def get_input_hw(itp):
    inp = itp.get_input_details()[0]
    return int(inp["shape"][1]), int(inp["shape"][2])

def preprocess_rgb_0_255(frame_bgr, target_hw):
    H, W = target_hw
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32)
    return np.expand_dims(x, 0)

def invoke_gate(itp, x):
    inp = itp.get_input_details()[0]
    out = itp.get_output_details()[0]
    itp.set_tensor(inp["index"], x)
    itp.invoke()
    return itp.get_tensor(out["index"])

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def to_prob(v):
    v = float(np.array(v).reshape(-1)[0])
    if v < 0.0 or v > 1.0:
        v = float(sigmoid(v))
    return v

# ---------- Camera ----------
def setup_camera():
    cap = cv2.VideoCapture(CAM_DEVICE, cv2.CAP_V4L2)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    if FORCE_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        return None
    return cap

# ---------- HSV mask ----------
def red_mask_hsv(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([H1_LOW,  S_MIN, V_MIN], dtype=np.uint8)
    upper1 = np.array([H1_HIGH, 255,   255  ], dtype=np.uint8)
    lower2 = np.array([H2_LOW,  S_MIN, V_MIN], dtype=np.uint8)
    upper2 = np.array([H2_HIGH, 255,   255  ], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    m = cv2.bitwise_or(m1, m2)

    k = np.ones((KERNEL, KERNEL), np.uint8)
    m = cv2.erode(m, k, iterations=ERODE_IT)
    m = cv2.dilate(m, k, iterations=DILATE_IT)
    return m

def red_ratio_in_box(mask, x, y, w, h):
    roi = mask[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / float(roi.size)

# ---------- UART ----------
def send_line(ser, s):
    ser.write((s + "\n").encode("ascii","ignore"))

def rline(ser):
    try:
        return ser.readline().decode("ascii","replace").strip()
    except:
        return ""

def wait_token(ser, token, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = rline(ser)
        if r == token:
            return True
    return False

def handshake(ser):
    if not wait_token(ser, "READY", timeout=5.0):
        return False
    send_line(ser, "HELLO")
    return wait_token(ser, "HELLO_OK", timeout=2.0)

def safe_stop(ser):
    try:
        send_line(ser, "PAN 0")
        send_line(ser, f"TILT {TILT_CENTER}")
    except:
        pass

# ---------- PAN pulse scheduler ----------
def pan_on_ms_from_err(err_px: float) -> int:
    """
    err_px: distancia al centro en pixeles (abs).
    Convierte error -> duración ON (ms).
    """
    a = abs(err_px)
    a = clamp(a, 0.0, float(PAN_ERR_FULL_PIX))
    t = PAN_ON_MIN_MS + (a / float(PAN_ERR_FULL_PIX)) * (PAN_ON_MAX_MS - PAN_ON_MIN_MS)
    return int(clamp(int(round(t)), PAN_ON_MIN_MS, PAN_ON_MAX_MS))

def main():
    # UART
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.02)
    time.sleep(0.2)
    ser.reset_input_buffer()

    if not handshake(ser):
        print("Handshake falló.")
        safe_stop(ser)
        return

    # CNN gate
    gate = make_interpreter(MODEL_PATH, threads=THREADS)
    gate_hw = get_input_hw(gate)

    # Camera
    cap = setup_camera()
    if cap is None:
        print("No se pudo abrir cámara.")
        safe_stop(ser)
        return

    # gate state
    p_s = 0.0
    active = False
    hold = 0

    # tilt state
    tilt_us_f = float(TILT_CENTER)

    # timing
    last_loop = time.time()
    last_send = 0.0
    last_seen = time.time()

    # PAN pulse state
    pan_state = "idle"  # idle | on | off
    pan_t_end = 0.0
    pan_last_dir = 0    # -1,0,+1
    pan_dir_lock_until = 0.0

    # UI stats
    t0 = time.time()
    frames = 0
    fps = 0.0
    frame_id = 0

    while True:
        # fps limiter
        now = time.time()
        dt_loop = now - last_loop
        min_dt = 1.0 / MAX_PROC_FPS
        if dt_loop < min_dt:
            time.sleep(min_dt - dt_loop)
        last_loop = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # fps calc
        frames += 1
        now = time.time()
        if now - t0 >= 1.0:
            fps = frames / (now - t0)
            frames = 0
            t0 = now

        Hf, Wf = frame.shape[:2]
        cx_frame = Wf * 0.5
        cy_frame = Hf * 0.5

        # ----- gating -----
        if (frame_id % PROCESS_EVERY) == 0:
            xg = preprocess_rgb_0_255(frame, gate_hw)
            p = to_prob(invoke_gate(gate, xg))
            p_s = (1 - ALPHA_GATE) * p_s + ALPHA_GATE * p

            if (not active) and p_s >= TH_ON:
                active = True
                hold = HOLD_FRAMES
            elif active and p_s <= TH_OFF:
                active = False

            if active:
                hold = HOLD_FRAMES
            if hold > 0:
                hold -= 1

        run_track = hold > 0

        # ----- blob detect -----
        dx_used = None
        dy_used = None
        err_px = None
        bbox = None
        centroid = None
        info = ""

        if run_track:
            mask = red_mask_hsv(frame)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if cnts:
                c = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(c)
                ar_frame = area / float(Wf * Hf)

                if area >= MIN_AREA and ar_frame <= MAX_AREA_FRAC:
                    x, y, w, h = cv2.boundingRect(c)
                    rr = red_ratio_in_box(mask, x, y, w, h)

                    if rr >= MIN_RED_RATIO:
                        cx = x + w / 2.0
                        cy = y + h / 2.0

                        dx_used = (cx - cx_frame) / cx_frame
                        dy_used = (cy - cy_frame) / cy_frame
                        err_px = (cx - cx_frame)

                        bbox = (x, y, w, h)
                        centroid = (int(cx), int(cy))
                        last_seen = time.time()

                        info = f"dx={dx_used:+.2f} dy={dy_used:+.2f} err_px={int(err_px)}"
                    else:
                        info = "reject rr"
                else:
                    info = "reject area/ar"
            else:
                info = "no blob"
        else:
            info = "SKIP (gate)"

        # ----- CONTROL -----
        now = time.time()

        # Lost target stop
        lost = (now - last_seen) > LOST_STOP_SEC
        if (dx_used is None) or (STOP_WHEN_LOST and lost):
            # stop pan pulses
            pan_state = "idle"
            if (now - last_send) >= send_dt:
                send_line(ser, "PAN 0")
                last_send = now
            # tilt queda donde estaba
        else:
            # ---------- PAN pulse control ----------
            # deadzone in pixels around crosshair
            if abs(err_px) <= PAN_DEAD_PIX:
                pan_state = "idle"
                pan_last_dir = 0
                if (now - last_send) >= send_dt:
                    send_line(ser, "PAN 0")
                    last_send = now
            else:
                # desired direction: err_px>0 => target right => rotate right
                desired_dir = 1 if err_px > 0 else -1
                desired_dir *= PAN_SIGN

                # anti-cabeceo: no cambies de dir cada frame
                if now < pan_dir_lock_until and pan_last_dir != 0:
                    desired_dir = pan_last_dir

                # schedule pulse
                if pan_state == "idle":
                    on_ms = pan_on_ms_from_err(err_px)
                    pan_t_end = now + (on_ms / 1000.0)
                    pan_state = "on"
                    pan_last_dir = desired_dir
                    pan_dir_lock_until = now + (PAN_DIR_COOLDOWN_MS / 1000.0)

                elif pan_state == "on":
                    if now >= pan_t_end:
                        pan_t_end = now + (PAN_OFF_MS / 1000.0)
                        pan_state = "off"

                elif pan_state == "off":
                    if now >= pan_t_end:
                        # next pulse
                        on_ms = pan_on_ms_from_err(err_px)
                        pan_t_end = now + (on_ms / 1000.0)
                        pan_state = "on"
                        # si cambió de lado, aplica lock
                        if desired_dir != pan_last_dir:
                            pan_last_dir = desired_dir
                            pan_dir_lock_until = now + (PAN_DIR_COOLDOWN_MS / 1000.0)

                # send PAN according to state
                pan_cmd_send = 0
                if pan_state == "on":
                    pan_cmd_send = pan_last_dir * PAN_PULSE_CMD
                else:
                    pan_cmd_send = 0
                # ---------- TILT (tu control suave) ----------
                if abs(dy_used) < TILT_DEADZONE:
                    target_tilt = TILT_CENTER
                else:
                    sign = -1.0 if TILT_INVERT else 1.0
                    target_tilt = TILT_CENTER + int(sign * KP_TILT_US * dy_used)

                target_tilt = int(clamp(target_tilt, TILT_MIN, TILT_MAX))
                tilt_us_f = (1 - TILT_SMOOTH) * tilt_us_f + TILT_SMOOTH * target_tilt

                # send rate-limited
                if (now - last_send) >= send_dt:
                    send_line(ser, f"PAN {int(pan_cmd_send)}")
                    send_line(ser, f"TILT {int(tilt_us_f)}")
                    last_send = now

        # ----- UI overlays -----
        # crosshair
        cv2.line(frame, (int(cx_frame), 0), (int(cx_frame), Hf), (0, 255, 0), 1)
        cv2.line(frame, (0, int(cy_frame)), (Wf, int(cy_frame)), (0, 255, 0), 1)
        cv2.circle(frame, (int(cx_frame), int(cy_frame)), 6, (0, 255, 0), 2)

        # bbox + centroid
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        if centroid is not None:
            cv2.circle(frame, centroid, 5, (0, 255, 255), -1)

        # deadzone box (visual)
        dz = PAN_DEAD_PIX
        cv2.rectangle(
            frame,
            (int(cx_frame - dz), int(cy_frame - dz)),
            (int(cx_frame + dz), int(cy_frame + dz)),
            (0, 255, 0),
            1,
        )

        cv2.putText(
            frame,
            f"FPS {fps:.1f} gate={p_s:.2f} {'TRACK' if run_track else 'SKIP'}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            info,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"PAN state={pan_state} cmd={PAN_PULSE_CMD} sign={PAN_SIGN} deadpx={PAN_DEAD_PIX} off={PAN_OFF_MS}ms",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"TILT={int(tilt_us_f)}",
            (10, 97),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.imshow("PAN/TILT tracker", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break

    # exit clean
    safe_stop(ser)
    cap.release()
    cv2.destroyAllWindows()
    ser.close()

if __name__ == "__main__":
    main()
                