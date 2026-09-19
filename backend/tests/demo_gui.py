import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.modules.nlp.pipeline import analyze_conversation
from backend.modules.risk.features import build_features
from backend.modules.risk.scorer import score_risk
from backend.modules.privacy.optimizer import optimize
from backend.modules.risk.features import build_features
from backend.modules.risk.scorer import score_risk
from backend.modules.privacy.optimizer import optimize


# ============================================================
# KAVACHAI — CONTEXTUAL PRIVACY GATEWAY
# Real M1 → M2 → M3 pipeline
# ============================================================


class KavachAIGUI:
    def __init__(self, root):
        self.root = root

        self.root.title(
            "KavachAI — Contextual Privacy Gateway"
        )
        self.root.geometry("1450x900")
        self.root.minsize(1100, 700)
        self.root.configure(bg="#faf8ff")

        # ----------------------------------------------------
        # COLORS
        # ----------------------------------------------------

        self.colors = {
            "primary": "#004ac6",
            "primary_dark": "#003b9f",
            "background": "#faf8ff",
            "surface": "#ffffff",
            "surface_low": "#f2f3ff",
            "surface_container": "#eaedff",
            "surface_high": "#e2e7ff",
            "text": "#131b2e",
            "text_secondary": "#434655",
            "muted": "#737686",
            "border": "#c3c6d7",
            "green": "#006243",
            "green_bg": "#e8f8f1",
            "purple": "#712ae2",
            "purple_bg": "#f1e9ff",
            "red": "#ba1a1a",
            "red_bg": "#ffdad6",
            "orange": "#a85c00",
            "orange_bg": "#fff2df",
        }

        self.fonts = {
            "title": ("Segoe UI", 19, "bold"),
            "section": ("Segoe UI", 13, "bold"),
            "body": ("Segoe UI", 10),
            "body_bold": ("Segoe UI", 10, "bold"),
            "small": ("Segoe UI", 9),
            "mono": ("Consolas", 9),
            "mono_bold": ("Consolas", 9, "bold"),
        }

        self.build_styles()
        self.build_ui()

    # ========================================================
    # STYLES
    # ========================================================

    def build_styles(self):
        style = ttk.Style()

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Treeview",
            background=self.colors["surface"],
            foreground=self.colors["text"],
            fieldbackground=self.colors["surface"],
            rowheight=36,
            font=self.fonts["small"],
            borderwidth=0,
        )

        style.configure(
            "Treeview.Heading",
            background=self.colors["surface_container"],
            foreground=self.colors["text_secondary"],
            font=("Segoe UI", 9, "bold"),
            padding=(8, 10),
        )

        style.map(
            "Treeview",
            background=[
                ("selected", "#dce8ff")
            ],
            foreground=[
                ("selected", self.colors["text"])
            ],
        )

    # ========================================================
    # MAIN UI
    # ========================================================

    def build_ui(self):

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header = tk.Frame(
            self.root,
            bg=self.colors["surface"],
            height=66,
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        header.pack(fill="x")
        header.pack_propagate(False)

        header_left = tk.Frame(
            header,
            bg=self.colors["surface"]
        )
        header_left.pack(
            side="left",
            padx=25,
            pady=12
        )

        shield = tk.Label(
            header_left,
            text="🛡",
            bg=self.colors["primary"],
            fg="white",
            font=("Segoe UI", 18, "bold"),
            width=3,
            height=1,
        )
        shield.pack(side="left", padx=(0, 10))

        title_frame = tk.Frame(
            header_left,
            bg=self.colors["surface"]
        )
        title_frame.pack(side="left")

        title_row = tk.Frame(
            title_frame,
            bg=self.colors["surface"]
        )
        title_row.pack(anchor="w")

        tk.Label(
            title_row,
            text="KavachAI",
            bg=self.colors["surface"],
            fg=self.colors["text"],
            font=self.fonts["title"],
        ).pack(side="left")

        tk.Label(
            title_row,
            text="  |  Contextual Privacy Gateway",
            bg=self.colors["surface"],
            fg=self.colors["text_secondary"],
            font=("Segoe UI", 10),
        ).pack(side="left")

        # Middleware status

        status_frame = tk.Frame(
            header,
            bg=self.colors["green_bg"],
            highlightbackground="#b8e7d2",
            highlightthickness=1,
        )
        status_frame.pack(
            side="right",
            padx=(10, 10),
            pady=17
        )

        tk.Label(
            status_frame,
            text="●",
            bg=self.colors["green_bg"],
            fg=self.colors["green"],
            font=("Segoe UI", 9),
        ).pack(side="left", padx=(10, 4))

        tk.Label(
            status_frame,
            text="MIDDLEWARE ACTIVE",
            bg=self.colors["green_bg"],
            fg=self.colors["green"],
            font=self.fonts["mono_bold"],
        ).pack(side="left", padx=(0, 10))

        reset_btn = tk.Button(
            header,
            text="↻  Reset View",
            command=self.reset_view,
            bg=self.colors["surface"],
            fg=self.colors["text_secondary"],
            activebackground=self.colors["surface_container"],
            relief="flat",
            bd=0,
            font=self.fonts["small"],
            cursor="hand2",
        )
        reset_btn.pack(
            side="right",
            padx=10
        )

        # ----------------------------------------------------
        # SCROLLABLE MAIN AREA
        # ----------------------------------------------------

        outer = tk.Frame(
            self.root,
            bg=self.colors["background"]
        )
        outer.pack(
            fill="both",
            expand=True
        )

        self.canvas = tk.Canvas(
            outer,
            bg=self.colors["background"],
            highlightthickness=0,
        )

        scrollbar = ttk.Scrollbar(
            outer,
            orient="vertical",
            command=self.canvas.yview
        )

        self.canvas.configure(
            yscrollcommand=scrollbar.set
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        self.canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        self.main_frame = tk.Frame(
            self.canvas,
            bg=self.colors["background"]
        )

        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.main_frame,
            anchor="nw"
        )

        self.main_frame.bind(
            "<Configure>",
            self.update_scroll_region
        )

        self.canvas.bind(
            "<Configure>",
            self.resize_canvas_content
        )

        self.canvas.bind_all(
            "<MouseWheel>",
            self.mousewheel
        )

        self.build_main_content()

    # ========================================================
    # MAIN CONTENT
    # ========================================================

    def build_main_content(self):

        container = tk.Frame(
            self.main_frame,
            bg=self.colors["background"]
        )
        container.pack(
            fill="both",
            expand=True,
            padx=28,
            pady=24
        )

        # ----------------------------------------------------
        # ORIGINAL MESSAGE + SANITIZED OUTPUT
        # ----------------------------------------------------

        top = tk.Frame(
            container,
            bg=self.colors["background"]
        )
        top.pack(
            fill="x"
        )

        top.grid_columnconfigure(
            0,
            weight=1
        )

        top.grid_columnconfigure(
            1,
            weight=1
        )

        self.build_input_panel(top)
        self.build_output_panel(top)

        # ----------------------------------------------------
        # RISK SUMMARY
        # ----------------------------------------------------

        self.build_risk_summary(container)

        # ----------------------------------------------------
        # CHANGED ENTITIES
        # ----------------------------------------------------

        self.build_entities_panel(container)

        # ----------------------------------------------------
        # PIPELINE
        # ----------------------------------------------------

        self.build_pipeline_panel(container)

        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------

        footer = tk.Frame(
            container,
            bg=self.colors["background"]
        )
        footer.pack(
            fill="x",
            pady=(20, 4)
        )

        tk.Label(
            footer,
            text="KavachAI Contextual Gateway",
            bg=self.colors["background"],
            fg=self.colors["text_secondary"],
            font=self.fonts["small"],
        ).pack(side="left")

        tk.Label(
            footer,
            text="M1 Detection  •  M2 Risk Scoring  •  M3 Selective Generalization",
            bg=self.colors["background"],
            fg=self.colors["muted"],
            font=self.fonts["small"],
        ).pack(side="right")

    # ========================================================
    # INPUT PANEL
    # ========================================================

    def build_input_panel(self, parent):

        panel = self.create_card(parent)

        panel.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 8)
        )

        header = tk.Frame(
            panel,
            bg=self.colors["surface"]
        )
        header.pack(
            fill="x",
            padx=18,
            pady=(16, 8)
        )

        tk.Label(
            header,
            text="↳",
            bg=self.colors["surface"],
            fg=self.colors["primary"],
            font=("Segoe UI", 17, "bold"),
        ).pack(side="left", padx=(0, 7))

        tk.Label(
            header,
            text="Original Message",
            bg=self.colors["surface"],
            fg=self.colors["text"],
            font=self.fonts["section"],
        ).pack(side="left")

        tk.Label(
            header,
            text="M1 INPUT",
            bg=self.colors["surface_container"],
            fg=self.colors["text_secondary"],
            font=self.fonts["mono_bold"],
            padx=8,
            pady=3,
        ).pack(side="right")

        self.message_input = ScrolledText(
            panel,
            height=11,
            wrap="word",
            font=("Segoe UI", 10),
            bg=self.colors["surface_low"],
            fg=self.colors["text"],
            insertbackground=self.colors["primary"],
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
        )

        self.message_input.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=(2, 12)
        )

        self.message_input.insert(
            "1.0",
            "Enter or paste text to analyze for sensitive data and contextual re-identification risk..."
        )

        self.message_input.bind(
            "<FocusIn>",
            self.clear_placeholder
        )

        # Character counter

        self.char_counter = tk.Label(
            panel,
            text="0 chars",
            bg=self.colors["surface"],
            fg=self.colors["muted"],
            font=self.fonts["mono"],
        )
        self.char_counter.pack(
            anchor="e",
            padx=20
        )

        self.message_input.bind(
            "<KeyRelease>",
            self.update_char_counter
        )

        # Buttons

        button_frame = tk.Frame(
            panel,
            bg=self.colors["surface"]
        )
        button_frame.pack(
            fill="x",
            padx=18,
            pady=14
        )

        self.analyze_btn = tk.Button(
            button_frame,
            text="✓  Analyze Message",
            command=self.analyze_message,
            bg=self.colors["primary"],
            fg="white",
            activebackground=self.colors["primary_dark"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=18,
            pady=9,
            font=self.fonts["body_bold"],
            cursor="hand2",
        )
        self.analyze_btn.pack(
            side="left"
        )

        clear_btn = tk.Button(
            button_frame,
            text="×  Clear",
            command=self.reset_view,
            bg=self.colors["surface_container"],
            fg=self.colors["text_secondary"],
            activebackground=self.colors["surface_high"],
            relief="flat",
            bd=0,
            padx=16,
            pady=9,
            font=self.fonts["body"],
            cursor="hand2",
        )
        clear_btn.pack(
            side="left",
            padx=8
        )

        tk.Label(
            button_frame,
            text="Ctrl + Enter to analyze",
            bg=self.colors["surface"],
            fg=self.colors["muted"],
            font=self.fonts["small"],
        ).pack(
            side="right"
        )

        self.message_input.bind(
            "<Control-Return>",
            lambda event: self.analyze_message()
        )

    # ========================================================
    # OUTPUT PANEL
    # ========================================================

    def build_output_panel(self, parent):

        panel = self.create_card(parent)

        panel.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(8, 0)
        )

        header = tk.Frame(
            panel,
            bg=self.colors["surface"]
        )
        header.pack(
            fill="x",
            padx=18,
            pady=(16, 8)
        )

        tk.Label(
            header,
            text="✓",
            bg=self.colors["surface"],
            fg=self.colors["green"],
            font=("Segoe UI", 17, "bold"),
        ).pack(side="left", padx=(0, 7))

        tk.Label(
            header,
            text="Sanitized Output",
            bg=self.colors["surface"],
            fg=self.colors["text"],
            font=self.fonts["section"],
        ).pack(side="left")

        self.output_status = tk.Label(
            header,
            text="AWAITING ANALYSIS",
            bg=self.colors["surface_container"],
            fg=self.colors["muted"],
            font=self.fonts["mono_bold"],
            padx=8,
            pady=3,
        )
        self.output_status.pack(
            side="right"
        )

        self.output_text = ScrolledText(
            panel,
            height=11,
            wrap="word",
            font=("Segoe UI", 10),
            bg=self.colors["surface_low"],
            fg=self.colors["text"],
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            state="disabled",
        )

        self.output_text.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=2
        )

        button_frame = tk.Frame(
            panel,
            bg=self.colors["surface"]
        )
        button_frame.pack(
            fill="x",
            padx=18,
            pady=14
        )

        self.copy_btn = tk.Button(
            button_frame,
            text="▣  Copy Sanitized Text",
            command=self.copy_output,
            bg=self.colors["surface_container"],
            fg=self.colors["muted"],
            activebackground=self.colors["surface_high"],
            relief="flat",
            bd=0,
            padx=14,
            pady=8,
            font=self.fonts["body"],
            state="disabled",
        )
        self.copy_btn.pack(
            side="left"
        )

        tk.Label(
            button_frame,
            text="✓  Zero-Trust LLM Safe",
            bg=self.colors["surface"],
            fg=self.colors["green"],
            font=self.fonts["small"],
        ).pack(
            side="right"
        )

    # ========================================================
    # RISK SUMMARY
    # ========================================================

    def build_risk_summary(self, parent):

        self.risk_panel = self.create_card(parent)

        self.risk_panel.pack(
            fill="x",
            pady=(16, 0)
        )

        header = tk.Frame(
            self.risk_panel,
            bg=self.colors["surface"]
        )
        header.pack(
            fill="x",
            padx=18,
            pady=(15, 8)
        )

        tk.Label(
            header,
            text="◈",
            bg=self.colors["surface"],
            fg=self.colors["purple"],
            font=("Segoe UI", 17, "bold"),
        ).pack(side="left", padx=(0, 7))

        tk.Label(
            header,
            text="Contextual Risk Analysis",
            bg=self.colors["surface"],
            fg=self.colors["text"],
            font=self.fonts["section"],
        ).pack(side="left")

        self.risk_level_label = tk.Label(
            header,
            text="NOT ANALYZED",
            bg=self.colors["surface_container"],
            fg=self.colors["muted"],
            font=self.fonts["mono_bold"],
            padx=9,
            pady=4,
        )
        self.risk_level_label.pack(
            side="right"
        )

        metrics = tk.Frame(
            self.risk_panel,
            bg=self.colors["surface"]
        )
        metrics.pack(
            fill="x",
            padx=18,
            pady=(4, 18)
        )

        self.initial_score_value = self.create_metric(
            metrics,
            "Initial Risk",
            "—"
        )
        self.initial_score_value.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 6)
        )

        self.final_score_value = self.create_metric(
            metrics,
            "Final Risk",
            "—"
        )
        self.final_score_value.pack(
            side="left",
            fill="x",
            expand=True,
            padx=6
        )

        self.risk_reduction_value = self.create_metric(
            metrics,
            "Risk Reduction",
            "—"
        )
        self.risk_reduction_value.pack(
            side="left",
            fill="x",
            expand=True,
            padx=6
        )

        self.utility_value = self.create_metric(
            metrics,
            "Utility Retained",
            "—"
        )
        self.utility_value.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(6, 0)
        )

    def create_metric(self, parent, title, value):

        frame = tk.Frame(
            parent,
            bg=self.colors["surface_low"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )

        tk.Label(
            frame,
            text=title,
            bg=self.colors["surface_low"],
            fg=self.colors["muted"],
            font=self.fonts["small"],
        ).pack(
            anchor="w",
            padx=12,
            pady=(9, 2)
        )

        value_label = tk.Label(
            frame,
            text=value,
            bg=self.colors["surface_low"],
            fg=self.colors["text"],
            font=("Segoe UI", 16, "bold"),
        )
        value_label.pack(
            anchor="w",
            padx=12,
            pady=(0, 9)
        )

        return value_label

    # ========================================================
    # ENTITIES TABLE
    # ========================================================

    def build_entities_panel(self, parent):

        panel = self.create_card(parent)

        panel.pack(
            fill="both",
            expand=True,
            pady=(16, 0)
        )

        header = tk.Frame(
            panel,
            bg=self.colors["surface"]
        )
        header.pack(
            fill="x",
            padx=18,
            pady=(15, 10)
        )

        tk.Label(
            header,
            text="↔",
            bg=self.colors["surface"],
            fg=self.colors["primary"],
            font=("Segoe UI", 17, "bold"),
        ).pack(side="left", padx=(0, 7))

        tk.Label(
            header,
            text="Changed Entities & Transformations",
            bg=self.colors["surface"],
            fg=self.colors["text"],
            font=self.fonts["section"],
        ).pack(side="left")

        self.entity_count_label = tk.Label(
            header,
            text="0 items",
            bg=self.colors["surface_container"],
            fg=self.colors["text_secondary"],
            font=self.fonts["mono_bold"],
            padx=8,
            pady=3,
        )
        self.entity_count_label.pack(
            side="left",
            padx=8
        )

        # Legend

        legend = tk.Frame(
            header,
            bg=self.colors["surface"]
        )
        legend.pack(
            side="right"
        )

        self.create_legend(
            legend,
            "Direct PII",
            self.colors["primary"]
        ).pack(
            side="left",
            padx=5
        )

        self.create_legend(
            legend,
            "Quasi-Identifier",
            self.colors["purple"]
        ).pack(
            side="left",
            padx=5
        )

        self.create_legend(
            legend,
            "Generalization",
            self.colors["green"]
        ).pack(
            side="left",
            padx=5
        )

        table_frame = tk.Frame(
            panel,
            bg=self.colors["surface"]
        )
        table_frame.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=(0, 18)
        )

        columns = (
            "original",
            "classification",
            "transformation",
            "transformed",
        )

        self.entities_table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=8,
        )

        self.entities_table.heading(
            "original",
            text="Original Entity"
        )

        self.entities_table.heading(
            "classification",
            text="Classification"
        )

        self.entities_table.heading(
            "transformation",
            text="Transformation Applied"
        )

        self.entities_table.heading(
            "transformed",
            text="Transformed Output"
        )

        self.entities_table.column(
            "original",
            width=240,
            anchor="w"
        )

        self.entities_table.column(
            "classification",
            width=160,
            anchor="center"
        )

        self.entities_table.column(
            "transformation",
            width=300,
            anchor="w"
        )

        self.entities_table.column(
            "transformed",
            width=250,
            anchor="w"
        )

        table_scroll = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.entities_table.yview
        )

        self.entities_table.configure(
            yscrollcommand=table_scroll.set
        )

        self.entities_table.pack(
            side="left",
            fill="both",
            expand=True
        )

        table_scroll.pack(
            side="right",
            fill="y"
        )

    def create_legend(self, parent, text, color):

        frame = tk.Frame(
            parent,
            bg=self.colors["surface"]
        )

        tk.Label(
            frame,
            text="●",
            bg=self.colors["surface"],
            fg=color,
            font=("Segoe UI", 9),
        ).pack(side="left")

        tk.Label(
            frame,
            text=text,
            bg=self.colors["surface"],
            fg=self.colors["text_secondary"],
            font=self.fonts["small"],
        ).pack(side="left", padx=2)

        return frame

    # ========================================================
    # PIPELINE
    # ========================================================

    def build_pipeline_panel(self, parent):

        panel = self.create_card(parent)

        panel.pack(
            fill="x",
            pady=(16, 0)
        )

        tk.Label(
            panel,
            text="KavachAI Middleware Pipeline",
            bg=self.colors["surface"],
            fg=self.colors["text"],
            font=self.fonts["section"],
        ).pack(
            anchor="w",
            padx=18,
            pady=(15, 12)
        )

        pipeline = tk.Frame(
            panel,
            bg=self.colors["surface"]
        )
        pipeline.pack(
            fill="x",
            padx=18,
            pady=(0, 18)
        )

        stages = [
            ("M1", "Detect & Extract", self.colors["primary"]),
            ("M2", "Contextual Risk", self.colors["purple"]),
            ("M3", "Selective Generalization", self.colors["green"]),
            ("✓", "Protected Data", self.colors["green"]),
        ]

        for index, (code, title, color) in enumerate(stages):

            stage = tk.Frame(
                pipeline,
                bg=self.colors["surface_low"],
                highlightbackground=self.colors["border"],
                highlightthickness=1,
            )

            stage.pack(
                side="left",
                fill="x",
                expand=True
            )

            tk.Label(
                stage,
                text=code,
                bg=color,
                fg="white",
                font=("Segoe UI", 10, "bold"),
                width=4,
                pady=5,
            ).pack(
                side="left",
                padx=8,
                pady=8
            )

            tk.Label(
                stage,
                text=title,
                bg=self.colors["surface_low"],
                fg=self.colors["text"],
                font=self.fonts["small"],
            ).pack(
                side="left"
            )

            if index < len(stages) - 1:

                tk.Label(
                    pipeline,
                    text="→",
                    bg=self.colors["surface"],
                    fg=self.colors["muted"],
                    font=("Segoe UI", 16, "bold"),
                ).pack(
                    side="left",
                    padx=8
                )

    # ========================================================
    # ANALYSIS
    # ========================================================

    def analyze_message(self):

        text = self.message_input.get(
            "1.0",
            tk.END
        ).strip()

        if not text:
            messagebox.showwarning(
                "Input Required",
                "Enter a message before running the analysis."
            )
            return

        self.analyze_btn.config(
            state="disabled",
            text="⟳  Analyzing..."
        )

        self.root.update_idletasks()

        try:

            # ------------------------------------------------
            # M1
            # ------------------------------------------------

            m1_result = analyze_conversation(
                conversation_id="gui_conversation",
                messages=[
                    {
                        "id": 1,
                        "text": text,
                    }
                ],
            )

            # ------------------------------------------------
            # M2
            # ------------------------------------------------

            features = build_features(
                m1_result
            )

            m2_result = score_risk(
                features
            )

            # ------------------------------------------------
            # M3
            # ------------------------------------------------

            m3_result = optimize(
                m1_result
            )

            # ------------------------------------------------
            # Render everything
            # ------------------------------------------------

            self.render_output(
                m1_result,
                m2_result,
                m3_result
            )

        except Exception as exc:

            messagebox.showerror(
                "KavachAI Analysis Error",
                f"The pipeline could not complete.\n\n{type(exc).__name__}: {exc}"
            )

        finally:

            self.analyze_btn.config(
                state="normal",
                text="✓  Analyze Message"
            )

    # ========================================================
    # RENDER RESULTS
    # ========================================================

    def render_output(
        self,
        m1_result,
        m2_result,
        m3_result
    ):

        # ----------------------------------------------------
        # Sanitized text
        # ----------------------------------------------------

        sanitized = m3_result.get(
            "sanitized_text",
            ""
        )

        self.output_text.config(
            state="normal"
        )

        self.output_text.delete(
            "1.0",
            tk.END
        )

        self.output_text.insert(
            "1.0",
            sanitized
        )

        self.output_text.config(
            state="disabled"
        )

        self.output_status.config(
            text="PROTECTED",
            bg=self.colors["green_bg"],
            fg=self.colors["green"]
        )

        self.copy_btn.config(
            state="normal",
            fg=self.colors["text"]
        )

        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        initial_score = m3_result.get(
            "initial_risk_score",
            m2_result.get("risk_score", 0)
        )

        final_score = m3_result.get(
            "final_risk_score",
            0
        )

        risk_reduction = m3_result.get(
            "risk_reduction",
            0
        )

        utility = m3_result.get(
            "utility_score",
            m3_result.get(
                "utility",
                0
            )
        )

        initial_level = m3_result.get(
            "initial_risk_level",
            m2_result.get("risk_level", "UNKNOWN")
        )

        final_level = m3_result.get(
            "final_risk_level",
            "UNKNOWN"
        )

        self.initial_score_value.config(
            text=f"{float(initial_score):.2f}"
        )

        self.final_score_value.config(
            text=f"{float(final_score):.2f}"
        )

        self.risk_reduction_value.config(
            text=f"{float(risk_reduction) * 100:.1f}%"
        )

        self.utility_value.config(
            text=f"{float(utility) * 100:.1f}%"
        )

        self.set_risk_level(
            final_level
        )

        # ----------------------------------------------------
        # Changed entities
        # ----------------------------------------------------

        self.render_entities(
            m3_result
        )

    # ========================================================
    # ENTITY RENDERING
    # ========================================================

    def render_entities(self, m3_result):

        for item in self.entities_table.get_children():

            self.entities_table.delete(
                item
            )

        transformations = m3_result.get(
            "transformations",
            []
        )

        self.entity_count_label.config(
            text=f"{len(transformations)} items"
        )

        for transformation in transformations:

            attribute = transformation.get(
                "attribute",
                "UNKNOWN"
            )

            original = transformation.get(
                "original_value",
                ""
            )

            transformed = transformation.get(
                "generalized_value",
                ""
            )

            specificity_before = transformation.get(
                "specificity_before",
                ""
            )

            specificity_after = transformation.get(
                "specificity_after",
                ""
            )

            classification = self.classification_for(
                attribute
            )

            transform_name = self.transformation_name(
                attribute,
                original,
                transformed,
                specificity_before,
                specificity_after
            )

            self.entities_table.insert(
                "",
                "end",
                values=(
                    original,
                    classification,
                    transform_name,
                    transformed,
                )
            )

    def classification_for(self, attribute):

        direct_pii = {
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "AADHAAR",
            "PAN",
            "BANK_ACCOUNT",
            "IFSC",
            "VEHICLE_NUMBER",
            "VOTER_ID",
            "PERSON",
        }

        quasi = {
            "AGE",
            "ADDRESS",
            "LOCATION",
            "DATE",
            "FACILITY",
        }

        if attribute in direct_pii:
            return "Direct PII"

        if attribute in quasi:
            return "Quasi-Identifier"

        if attribute == "UNIQUENESS":
            return "Generalization"

        return "Generalization"

    def transformation_name(
        self,
        attribute,
        original,
        transformed,
        specificity_before,
        specificity_after
    ):

        if attribute == "UNIQUENESS":
            return "Contextual uniqueness reduction"

        if original == transformed:
            return "No transformation"

        if (
            specificity_before != ""
            and specificity_after != ""
        ):

            try:

                before = float(
                    specificity_before
                )

                after = float(
                    specificity_after
                )

                if after < before:
                    return "Specificity reduction"

            except (
                ValueError,
                TypeError
            ):
                pass

        direct_pii = {
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "AADHAAR",
            "PAN",
            "BANK_ACCOUNT",
            "IFSC",
            "VEHICLE_NUMBER",
            "VOTER_ID",
            "PERSON",
        }

        if attribute in direct_pii:
            return "Direct identifier protection"

        if attribute == "AGE":
            return "Age generalization"

        if attribute in {
            "ADDRESS",
            "LOCATION"
        }:
            return "Geographic generalization"

        if attribute == "HEALTH":
            return "Health information generalization"

        if attribute == "EDUCATION":
            return "Education generalization"

        return "Contextual generalization"

    # ========================================================
    # RISK LEVEL
    # ========================================================

    def set_risk_level(self, level):

        level = str(
            level
        ).upper()

        if level == "HIGH":

            bg = self.colors["red_bg"]
            fg = self.colors["red"]

        elif level == "MEDIUM":

            bg = self.colors["orange_bg"]
            fg = self.colors["orange"]

        elif level == "LOW":

            bg = self.colors["green_bg"]
            fg = self.colors["green"]

        else:

            bg = self.colors["surface_container"]
            fg = self.colors["muted"]

        self.risk_level_label.config(
            text=level,
            bg=bg,
            fg=fg
        )

    # ========================================================
    # CLEAR / RESET
    # ========================================================

    def reset_view(self):

        self.message_input.delete(
            "1.0",
            tk.END
        )

        self.update_char_counter()

        self.output_text.config(
            state="normal"
        )

        self.output_text.delete(
            "1.0",
            tk.END
        )

        self.output_text.config(
            state="disabled"
        )

        self.output_status.config(
            text="AWAITING ANALYSIS",
            bg=self.colors["surface_container"],
            fg=self.colors["muted"]
        )

        self.copy_btn.config(
            state="disabled"
        )

        self.initial_score_value.config(
            text="—"
        )

        self.final_score_value.config(
            text="—"
        )

        self.risk_reduction_value.config(
            text="—"
        )

        self.utility_value.config(
            text="—"
        )

        self.risk_level_label.config(
            text="NOT ANALYZED",
            bg=self.colors["surface_container"],
            fg=self.colors["muted"]
        )

        for item in self.entities_table.get_children():

            self.entities_table.delete(
                item
            )

        self.entity_count_label.config(
            text="0 items"
        )

        self.message_input.focus()

    # ========================================================
    # COPY
    # ========================================================

    def copy_output(self):

        self.output_text.config(
            state="normal"
        )

        text = self.output_text.get(
            "1.0",
            tk.END
        ).strip()

        self.output_text.config(
            state="disabled"
        )

        if not text:
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(
            text
        )
        self.root.update()

        self.copy_btn.config(
            text="✓  Copied!"
        )

        self.root.after(
            1800,
            lambda: self.copy_btn.config(
                text="▣  Copy Sanitized Text"
            )
        )

    # ========================================================
    # HELPERS
    # ========================================================

    def create_card(self, parent):

        return tk.Frame(
            parent,
            bg=self.colors["surface"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            bd=0,
        )

    def clear_placeholder(self, event=None):

        current = self.message_input.get(
            "1.0",
            tk.END
        ).strip()

        if current.startswith(
            "Enter or paste text to analyze"
        ):

            self.message_input.delete(
                "1.0",
                tk.END
            )

    def update_char_counter(self, event=None):

        text = self.message_input.get(
            "1.0",
            tk.END
        ).rstrip("\n")

        self.char_counter.config(
            text=f"{len(text)} chars"
        )

    def update_scroll_region(self, event=None):

        self.canvas.configure(
            scrollregion=self.canvas.bbox("all")
        )

    def resize_canvas_content(self, event):

        self.canvas.itemconfigure(
            self.canvas_window,
            width=event.width
        )

    def mousewheel(self, event):

        self.canvas.yview_scroll(
            int(-1 * (event.delta / 120)),
            "units"
        )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = KavachAIGUI(
        root
    )

    root.mainloop()