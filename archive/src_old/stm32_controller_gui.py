# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox
import serial
import serial.tools.list_ports
import time


class STM32PlatformController:
    def __init__(self, root):
        self.root = root
        self.root.title("3축 플랫폼 STM32 제어기 (Ubuntu)")

        # 창 크기 설정
        self.root.geometry("600x720")
        self.root.minsize(550, 650)

        # 스타일 설정
        self.style = ttk.Style()
        self.style.theme_use("clam")

        default_font = ("Noto Sans CJK KR", 11)
        header_font = ("Noto Sans CJK KR", 12, "bold")
        btn_font = ("Noto Sans CJK KR", 11, "bold")

        self.style.configure(".", font=default_font)
        self.style.configure("TLabelframe.Label", font=header_font)
        self.style.configure("TButton", font=btn_font, padding=6)
        self.style.configure(
            "Accent.TButton",
            font=("Noto Sans CJK KR", 12, "bold"),
            padding=8
        )

        self.ser = None

        self.create_widgets()
        self.refresh_com_ports()

    def create_widgets(self):

        # =========================================================
        # 1. 시리얼 포트 연결
        # =========================================================
        conn_frame = ttk.LabelFrame(
            self.root,
            text=" 1. 시리얼 포트 연결 ",
            padding=12
        )
        conn_frame.pack(fill="x", padx=15, pady=8)

        ttk.Label(
            conn_frame,
            text="포트 선택:"
        ).grid(row=0, column=0, sticky="w", padx=5)

        self.port_combo = ttk.Combobox(
            conn_frame,
            width=22,
            state="readonly",
            font=("Noto Sans CJK KR", 11)
        )
        self.port_combo.grid(
            row=0,
            column=1,
            padx=8,
            ipady=3
        )

        ttk.Button(
            conn_frame,
            text="새로고침",
            width=9,
            command=self.refresh_com_ports
        ).grid(
            row=0,
            column=2,
            padx=4
        )

        self.btn_connect = ttk.Button(
            conn_frame,
            text="연결",
            width=9,
            command=self.toggle_connection
        )
        self.btn_connect.grid(
            row=0,
            column=3,
            padx=4
        )

        # =========================================================
        # 2. 목표 제어값 입력
        # =========================================================
        ctrl_frame = ttk.LabelFrame(
            self.root,
            text=" 2. 목표 제어값 입력 (Z: cm / Roll, Pitch: deg) ",
            padding=12
        )
        ctrl_frame.pack(
            fill="x",
            padx=15,
            pady=8
        )

        # Z축
        ttk.Label(
            ctrl_frame,
            text="Z축 높이 (cm):"
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=8,
            padx=5
        )

        self.entry_z = ttk.Entry(
            ctrl_frame,
            width=15,
            font=("Noto Sans CJK KR", 12)
        )
        self.entry_z.insert(0, "0.0")
        self.entry_z.grid(
            row=0,
            column=1,
            pady=8,
            ipady=4
        )

        # Roll
        ttk.Label(
            ctrl_frame,
            text="Roll 각도 (deg):"
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=8,
            padx=5
        )

        self.entry_roll = ttk.Entry(
            ctrl_frame,
            width=15,
            font=("Noto Sans CJK KR", 12)
        )
        self.entry_roll.insert(0, "0.0")
        self.entry_roll.grid(
            row=1,
            column=1,
            pady=8,
            ipady=4
        )

        # Pitch
        ttk.Label(
            ctrl_frame,
            text="Pitch 각도 (deg):"
        ).grid(
            row=2,
            column=0,
            sticky="w",
            pady=8,
            padx=5
        )

        self.entry_pitch = ttk.Entry(
            ctrl_frame,
            width=15,
            font=("Noto Sans CJK KR", 12)
        )
        self.entry_pitch.insert(0, "0.0")
        self.entry_pitch.grid(
            row=2,
            column=1,
            pady=8,
            ipady=4
        )

        # 제어 명령 전송 버튼
        self.btn_send = ttk.Button(
            ctrl_frame,
            text="▶ 제어 명령 전송 (Send)",
            style="Accent.TButton",
            command=self.send_command
        )
        self.btn_send.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(12, 6)
        )

        # 원점 복귀 버튼
        self.btn_home = ttk.Button(
            ctrl_frame,
            text="수평 원점 복귀 (Z:0 R:0 P:0)",
            command=self.send_zero_pose
        )
        self.btn_home.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=4
        )

        # =========================================================
        # 3. 통신 로그
        # =========================================================
        log_frame = ttk.LabelFrame(
            self.root,
            text=" 3. 통신 로그 ",
            padding=10
        )
        log_frame.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=8
        )

        self.log_text = tk.Text(
            log_frame,
            height=8,
            font=("Noto Sans Mono CJK KR", 10),
            state="disabled"
        )
        self.log_text.pack(
            fill="both",
            expand=True
        )

    def log(self, message):
        self.log_text.config(state="normal")

        self.log_text.insert(
            "end",
            f"[{time.strftime('%H:%M:%S')}] {message}\n"
        )

        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def refresh_com_ports(self):
        ports = [
            port.device
            for port in serial.tools.list_ports.comports()
        ]

        self.port_combo["values"] = ports

        if ports:
            self.port_combo.current(0)

        self.log(
            f"포트 목록 갱신 완료 ({len(ports)}개 발견)"
        )

    def toggle_connection(self):

        # 현재 연결된 상태라면 연결 해제
        if self.ser and self.ser.is_open:

            self.ser.close()
            self.ser = None

            self.btn_connect.config(
                text="연결"
            )

            self.log(
                "시리얼 연결 해제됨"
            )

        # 연결되어 있지 않다면 연결
        else:

            port = self.port_combo.get()

            if not port:
                messagebox.showwarning(
                    "경고",
                    "포트를 선택해주세요."
                )
                return

            try:
                self.ser = serial.Serial(
                    port=port,
                    baudrate=115200,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=1,
                    write_timeout=1
                )

                self.btn_connect.config(
                    text="연결 해제"
                )

                self.log(
                    f"{port} 포트에 115200 bps로 연결 성공"
                )

            except Exception as e:

                messagebox.showerror(
                    "연결 오류",
                    f"포트 열기 실패:\n{str(e)}"
                )

    def send_raw_command(self, cmd_str):

        if not self.ser or not self.ser.is_open:

            messagebox.showwarning(
                "경고",
                "시리얼 포트가 연결되어 있지 않습니다."
            )

            return

        try:

            full_packet = (
                f"{cmd_str}\r\n"
            ).encode("utf-8")

            self.ser.write(
                full_packet
            )

            self.log(
                f">> 전송: {cmd_str}"
            )

        except Exception as e:

            self.log(
                f"전송 실패: {str(e)}"
            )

    def send_command(self):

        try:

            z = float(
                self.entry_z.get()
            )

            r = float(
                self.entry_roll.get()
            )

            p = float(
                self.entry_pitch.get()
            )

            cmd = (
                f"Z:{z:.2f} "
                f"R:{r:.2f} "
                f"P:{p:.2f}"
            )

            self.send_raw_command(
                cmd
            )

        except ValueError:

            messagebox.showerror(
                "입력 오류",
                "Z, Roll, Pitch에는 숫자만 입력해야 합니다."
            )

    def send_zero_pose(self):

        self.entry_z.delete(0, "end")
        self.entry_z.insert(0, "0.0")

        self.entry_roll.delete(0, "end")
        self.entry_roll.insert(0, "0.0")

        self.entry_pitch.delete(0, "end")
        self.entry_pitch.insert(0, "0.0")

        self.send_raw_command(
            "Z:0.00 R:0.00 P:0.00"
        )


if __name__ == "__main__":

    root = tk.Tk()

    app = STM32PlatformController(root)

    root.mainloop()
