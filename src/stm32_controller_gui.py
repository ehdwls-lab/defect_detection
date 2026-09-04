import tkinter as tk
from tkinter import ttk, messagebox
import serial
import serial.tools.list_ports
import time
import threading

MOTOR_LAYOUT = {
    1: (0.50, 0.20, "1 (앞)"),
    2: (0.22, 0.78, "2 (좌후)"),
    3: (0.78, 0.78, "3 (우후)"),
}


class STM32PlatformController:
    def __init__(self, root):
        self.root = root
        self.root.title("3축 플랫폼 STM32 제어기")
        self.root.geometry("660x820")
        self.root.minsize(620, 760)

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure(".", font=("Sans", 11))
        self.style.configure("TLabelframe.Label", font=("Sans", 12, "bold"))
        self.style.configure("TButton", font=("Sans", 11, "bold"), padding=6)
        self.style.configure("Accent.TButton", font=("Sans", 12, "bold"), padding=8)
        self.style.configure("Tlm.TLabel", font=("Monospace", 14, "bold"), foreground="#0055ff")

        self.ser = None
        self.rx_thread = None
        self.running = True
        self.motor_val = {1: 0, 2: 0, 3: 0}
        self.homing = False

        self.create_widgets()
        self.refresh_com_ports()
        self.draw_layout()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        conn_frame = ttk.LabelFrame(self.root, text=" 1. 시리얼 포트 연결 ", padding=12)
        conn_frame.pack(fill="x", padx=15, pady=6)

        ttk.Label(conn_frame, text="포트 선택:").grid(row=0, column=0, sticky="w", padx=5)
        self.port_combo = ttk.Combobox(conn_frame, width=22, state="readonly", font=("Sans", 11))
        self.port_combo.grid(row=0, column=1, padx=8, ipady=3)

        ttk.Button(conn_frame, text="새로고침", width=9,
                   command=self.refresh_com_ports).grid(row=0, column=2, padx=4)

        self.btn_connect = ttk.Button(conn_frame, text="연결", width=9, command=self.toggle_connection)
        self.btn_connect.grid(row=0, column=3, padx=4)

        mid = ttk.Frame(self.root)
        mid.pack(fill="x", padx=15, pady=6)

        tlm_frame = ttk.LabelFrame(mid, text=" 2. 실시간 상태 ", padding=12)
        tlm_frame.pack(side="left", fill="both", expand=True)

        ttk.Label(tlm_frame, text="Z 높이 :").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.lbl_cur_z = ttk.Label(tlm_frame, text="0.00 cm", style="Tlm.TLabel")
        self.lbl_cur_z.grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(tlm_frame, text="Roll φ (좌우) :").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.lbl_cur_roll = ttk.Label(tlm_frame, text="+0.00 deg", style="Tlm.TLabel")
        self.lbl_cur_roll.grid(row=1, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(tlm_frame, text="Pitch θ (앞뒤):").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.lbl_cur_pitch = ttk.Label(tlm_frame, text="+0.00 deg", style="Tlm.TLabel")
        self.lbl_cur_pitch.grid(row=2, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(tlm_frame, text="동작   :").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.lbl_status = ttk.Label(tlm_frame, text="연결 안됨",
                                    font=("Sans", 11, "bold"), foreground="gray")
        self.lbl_status.grid(row=3, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(tlm_frame, text="IMU    :").grid(row=4, column=0, sticky="w", padx=6, pady=4)
        self.lbl_imu = ttk.Label(tlm_frame, text="-",
                                 font=("Sans", 10), foreground="gray")
        self.lbl_imu.grid(row=4, column=1, sticky="w", padx=6, pady=4)

        diag_frame = ttk.LabelFrame(mid, text=" 모터 배치 (카메라 기준) ", padding=8)
        diag_frame.pack(side="left", fill="both", padx=(10, 0))

        self.canvas = tk.Canvas(diag_frame, width=230, height=200, highlightthickness=0)
        self.canvas.pack()

        ctrl_frame = ttk.LabelFrame(self.root, text=" 3. 목표 자세 (Z: cm / φ, θ: deg) ", padding=12)
        ctrl_frame.pack(fill="x", padx=15, pady=6)

        ttk.Label(ctrl_frame, text="목표 Z축 (cm):").grid(row=0, column=0, sticky="w", pady=6, padx=5)
        self.entry_z = ttk.Entry(ctrl_frame, width=15, font=("Sans", 12))
        self.entry_z.insert(0, "0.0")
        self.entry_z.grid(row=0, column=1, pady=6, ipady=3)

        ttk.Label(ctrl_frame, text="목표 Roll φ (deg):").grid(row=1, column=0, sticky="w", pady=6, padx=5)
        self.entry_roll = ttk.Entry(ctrl_frame, width=15, font=("Sans", 12))
        self.entry_roll.insert(0, "0.0")
        self.entry_roll.grid(row=1, column=1, pady=6, ipady=3)
        ttk.Label(ctrl_frame, text="좌우 기울기 / + : 3번(우) 상승, 2번(좌) 하강",
                  foreground="#555").grid(row=1, column=2, sticky="w", padx=10)

        ttk.Label(ctrl_frame, text="목표 Pitch θ (deg):").grid(row=2, column=0, sticky="w", pady=6, padx=5)
        self.entry_pitch = ttk.Entry(ctrl_frame, width=15, font=("Sans", 12))
        self.entry_pitch.insert(0, "0.0")
        self.entry_pitch.grid(row=2, column=1, pady=6, ipady=3)
        ttk.Label(ctrl_frame, text="앞뒤 기울기 / + : 1번(앞) 상승, 2·3번 하강",
                  foreground="#555").grid(row=2, column=2, sticky="w", padx=10)

        self.btn_send = ttk.Button(ctrl_frame, text="▶ 제어 명령 전송",
                                   style="Accent.TButton", command=self.send_command)
        self.btn_send.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 3))

        self.btn_home = ttk.Button(ctrl_frame, text="수평 원점 복귀 (Z:0 R:0 P:0)",
                                   command=self.send_zero_pose)
        self.btn_home.grid(row=4, column=0, columnspan=3, sticky="ew", pady=3)

        self.btn_reset = ttk.Button(ctrl_frame, text="원점 재설정 (하강 후 영점 재계산, 약 15초)",
                                    command=self.send_reset)
        self.btn_reset.grid(row=5, column=0, columnspan=3, sticky="ew", pady=3)

        log_frame = ttk.LabelFrame(self.root, text=" 4. 통신 로그 ", padding=10)
        log_frame.pack(fill="both", expand=True, padx=15, pady=6)

        self.log_text = tk.Text(log_frame, height=6, font=("Monospace", 10), state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def draw_layout(self):
        c = self.canvas
        c.delete("all")
        w, h, r = 230, 200, 26

        c.create_oval(w * 0.5 - 78, h * 0.5 - 68, w * 0.5 + 78, h * 0.5 + 68,
                      outline="#cccccc", dash=(3, 3))
        c.create_text(w * 0.5, 12, text="카메라 전방", fill="#888", font=("Sans", 9))

        for num, (fx, fy, label) in MOTOR_LAYOUT.items():
            x, y = fx * w, fy * h
            v = self.motor_val.get(num, 0)
            if v > 0:
                fill = "#2f7de1"
            elif v < 0:
                fill = "#e05a2f"
            else:
                fill = "#e8e8e8"
            c.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline="#555", width=2)
            c.create_text(x, y - 6, text=str(num), font=("Sans", 13, "bold"),
                          fill="white" if v != 0 else "#333")
            c.create_text(x, y + 10, text=str(v), font=("Monospace", 8),
                          fill="white" if v != 0 else "#666")
            c.create_text(x, y + r + 11, text=label, font=("Sans", 8), fill="#555")

    def log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def refresh_com_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            acm_ports = [p for p in ports if "ttyACM" in p]
            self.port_combo.set(acm_ports[0] if acm_ports else ports[0])
        self.log(f"포트 목록 갱신 완료 ({len(ports)}개 발견)")

    def toggle_connection(self):
        if self.ser and self.ser.is_open:
            self.running = False
            if self.rx_thread and self.rx_thread.is_alive():
                self.rx_thread.join(timeout=0.5)
            self.ser.close()
            self.ser = None
            self.btn_connect.config(text="연결")
            self.lbl_status.config(text="연결 해제됨", foreground="gray")
            self.log("시리얼 연결 해제됨")
        else:
            port = self.port_combo.get()
            if not port:
                messagebox.showwarning("경고", "포트를 선택해주세요.")
                return
            try:
                self.ser = serial.Serial(
                    port=port,
                    baudrate=115200,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=0.5,
                    write_timeout=1
                )
                self.btn_connect.config(text="연결 해제")
                self.lbl_status.config(text="연결됨 (수신 대기)", foreground="green")
                self.log(f"{port} 포트에 115200 bps로 연결 성공")

                self.running = True
                self.rx_thread = threading.Thread(target=self.receive_loop, daemon=True)
                self.rx_thread.start()
            except Exception as e:
                messagebox.showerror("연결 오류", f"포트 열기 실패:\n{str(e)}")

    def receive_loop(self):
        while self.running and self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("TLM:"):
                    data = {}
                    for p in line.replace("TLM:", "").split(","):
                        if "=" not in p:
                            continue
                        k, v = p.split("=", 1)
                        data[k] = v
                    self.root.after(0, self.update_dashboard, data)
            except Exception:
                pass
            time.sleep(0.01)

    def update_dashboard(self, data):
        try:
            z = float(data.get("Z", 0.0))
            r = float(data.get("R", 0.0))
            p = float(data.get("P", 0.0))
            s = int(float(data.get("S", 0)))
        except ValueError:
            return

        self.lbl_cur_z.config(text=f"{z:.2f} cm")
        self.lbl_cur_roll.config(text=f"{r:+.2f} deg")
        self.lbl_cur_pitch.config(text=f"{p:+.2f} deg")

        for n in (1, 2, 3):
            key = f"M{n}"
            if key in data:
                try:
                    self.motor_val[n] = int(float(data[key]))
                except ValueError:
                    self.motor_val[n] = 0
        self.draw_layout()

        try:
            self.homing = int(float(data.get("H", 0))) == 1
        except ValueError:
            self.homing = False

        if "G" in data:
            try:
                g = int(float(data["G"]))
            except ValueError:
                g = 0
            if g == 1:
                self.lbl_imu.config(text="자이로 미분", foreground="#00aa00")
            elif g == 2:
                self.lbl_imu.config(text="각도 차분 (자이로 실패)", foreground="#cc6600")
            else:
                self.lbl_imu.config(text="판별 중", foreground="gray")

        for b in (self.btn_send, self.btn_home, self.btn_reset):
            b.config(state="disabled" if self.homing else "normal")

        if self.homing:
            self.lbl_status.config(text="◆ 원점 재설정 중", foreground="#cc00cc")
        elif s == 1:
            self.lbl_status.config(text="● 목표 도달 (대기)", foreground="#00aa00")
        else:
            self.lbl_status.config(text="▶ 제어 중", foreground="#ff6600")

    def send_reset(self):
        if not self.ser or not self.ser.is_open:
            messagebox.showwarning("경고", "시리얼 포트가 연결되어 있지 않습니다.")
            return
        if not messagebox.askokcancel(
                "원점 재설정",
                "세 실린더를 최하단까지 내린 뒤 IMU 영점을 다시 계산합니다.\n"
                "약 15초 동안 자동 제어가 정지됩니다. 진행할까요?"):
            return
        self.send_raw_command("RST")

    def send_raw_command(self, cmd_str):
        if not self.ser or not self.ser.is_open:
            messagebox.showwarning("경고", "시리얼 포트가 연결되어 있지 않습니다.")
            return
        try:
            self.ser.write(f"{cmd_str}\r\n".encode('utf-8'))
            self.log(f">> 전송: {cmd_str}")
        except Exception as e:
            self.log(f"전송 실패: {str(e)}")

    def send_command(self):
        try:
            z = float(self.entry_z.get())
            r = float(self.entry_roll.get())
            p = float(self.entry_pitch.get())
            self.send_raw_command(f"Z:{z:.2f} R:{r:.2f} P:{p:.2f}")
        except ValueError:
            messagebox.showerror("입력 오류", "Z, φ, θ에는 숫자만 입력해야 합니다.")

    def send_zero_pose(self):
        self.entry_z.delete(0, "end"); self.entry_z.insert(0, "0.0")
        self.entry_roll.delete(0, "end"); self.entry_roll.insert(0, "0.0")
        self.entry_pitch.delete(0, "end"); self.entry_pitch.insert(0, "0.0")
        self.send_raw_command("Z:0.00 R:0.00 P:0.00")

    def on_closing(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = STM32PlatformController(root)
    root.mainloop()