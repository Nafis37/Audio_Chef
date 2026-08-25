"""
Audio Chef — Day 1: Audio as a Signal

A WAV file is just a NumPy array of amplitude numbers, captured
`sample_rate` times a second. This app loads a file, shows you that
array's real shape and stats, lets you play it back, and plots the
waveform — mono as one line, stereo as left/right channels stacked.

Both the UI and the audio analysis are plain Python (NiceGUI serves the
frontend; soundfile/numpy/plotly do the signal work).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nicegui import app, events, ui

from audio_core import SignalInfo, analyze, downsample_for_plot, load_audio

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
app.add_static_files("/uploads", str(UPLOAD_DIR))

BG = "#0b0d12"
SURFACE = "#12151c"
SURFACE_2 = "#171b25"
BORDER = "#242938"
TEXT = "#e7e9ee"
MUTED = "#8b93a7"
ACCENT = "#22d3ee"
ACCENT_2 = "#a78bfa"


def build_waveform_figure(data, sr: int) -> go.Figure:
    """Interactive waveform plot. One row per channel, x-axis in seconds
    (not sample index) so the sample-rate concept is visible in the plot."""

    is_stereo = data.ndim > 1
    channel_count = data.shape[1] if is_stereo else 1
    labels = ["Left channel", "Right channel"] if is_stereo else ["Mono"]
    colors = [ACCENT, ACCENT_2]
    fill_colors = ["rgba(34, 211, 238, 0.13)", "rgba(167, 139, 250, 0.13)"]

    fig = make_subplots(
        rows=channel_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=labels[:channel_count],
    )

    for ch in range(channel_count):
        channel_data = data[:, ch] if is_stereo else data
        xs, ys = downsample_for_plot(channel_data)
        fig.add_trace(
            go.Scatter(
                x=xs / sr,
                y=ys,
                mode="lines",
                line=dict(color=colors[ch % 2], width=1),
                fill="tozeroy",
                fillcolor=fill_colors[ch % 2],
                hovertemplate="t = %{x:.3f}s<br>amplitude = %{y:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=ch + 1,
            col=1,
        )
        fig.update_yaxes(
            title_text="Amplitude", range=[-1.05, 1.05], row=ch + 1, col=1,
            gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED,
        )

    fig.update_xaxes(
        title_text="Time (s)", row=channel_count, col=1,
        gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED,
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED)
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT, family="Plus Jakarta Sans, sans-serif"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=230 * channel_count,
    )
    for annotation in fig.layout.annotations:
        annotation.font = dict(color=MUTED, size=13)
    return fig


def stat_card(icon: str, label: str, value: str, sub: str = "") -> None:
    with ui.column().classes(
        "flex-1 min-w-[160px] rounded-2xl p-4 gap-1"
    ).style(f"background:{SURFACE}; border:1px solid {BORDER};"):
        with ui.row().classes("items-center gap-2"):
            ui.icon(icon).style(f"color:{ACCENT}; font-size:1.15rem;")
            ui.label(label).style(f"color:{MUTED}; font-size:0.78rem; letter-spacing:.04em; text-transform:uppercase;")
        ui.label(value).style(f"color:{TEXT}; font-size:1.45rem; font-weight:700;")
        if sub:
            ui.label(sub).style(f"color:{MUTED}; font-size:0.8rem;")


@ui.page("/")
def main_page() -> None:
    ui.add_head_html(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        """
    )
    ui.query("body").style(f"background:{BG};")
    ui.add_css(
        f"""
        * {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
        .mono {{ font-family: 'JetBrains Mono', monospace; }}
        .material-icons {{
            font-family: 'Material Icons';
            font-weight: normal;
            font-style: normal;
            line-height: 1;
            letter-spacing: normal;
            text-transform: none;
            white-space: nowrap;
            word-wrap: normal;
            direction: ltr;
            -webkit-font-feature-settings: 'liga';
            -webkit-font-smoothing: antialiased;
        }}
        .q-uploader {{ background: transparent !important; box-shadow: none !important; }}
        ::-webkit-scrollbar {{ height: 8px; width: 8px; }}
        ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 8px; }}
        """
    )

    result_area: ui.column

    def render_placeholder() -> None:
        result_area.clear()
        with result_area:
            with ui.column().classes("w-full items-center justify-center py-16 gap-3").style(
                f"border:1px dashed {BORDER}; border-radius:1.25rem;"
            ):
                ui.icon("graphic_eq").style(f"color:{MUTED}; font-size:2.75rem;")
                ui.label("Upload a WAV file to see it as a signal").style(
                    f"color:{MUTED}; font-size:1rem;"
                )

    def render_results(path: Path, original_name: str) -> None:
        try:
            data, sr = load_audio(str(path))
            info: SignalInfo = analyze(data, sr, str(path))
        except Exception as exc:  # noqa: BLE001 — surfaced to the user, not swallowed
            ui.notify(f"Couldn't read that file as audio: {exc}", type="negative")
            return

        result_area.clear()
        with result_area:
            ui.label(f"🎧 {original_name}").style(
                f"color:{TEXT}; font-size:1.1rem; font-weight:600;"
            )

            ui.audio(f"/uploads/{path.name}").classes("w-full").style(
                f"border-radius:0.75rem;"
            )

            with ui.element("div").classes("w-full gap-3").style(
                "display:grid; grid-template-columns:repeat(3, minmax(0,1fr));"
            ):
                stat_card("speed", "Sample Rate", f"{info.sample_rate:,} Hz",
                          "samples captured per second")
                stat_card("schedule", "Duration", f"{info.duration_sec:.3f} s")
                stat_card(
                    "stereo" if info.channels > 1 else "hearing",
                    "Channels",
                    "Stereo" if info.channels > 1 else "Mono",
                    f"{info.channels} channel(s)",
                )
                stat_card("grid_4x4", "Total Samples", f"{info.num_samples:,}",
                          "array length, per channel")
                stat_card("tune", "Bit Depth / Encoding", info.subtype)
                stat_card("show_chart", "Amplitude Range",
                           f"{info.min_amplitude:.3f} to {info.max_amplitude:.3f}",
                           f"RMS {info.rms_amplitude:.3f}")

            ui.label("Waveform").style(f"color:{TEXT}; font-size:1rem; font-weight:600; margin-top:0.5rem;")
            fig = build_waveform_figure(data, sr)
            ui.plotly(fig).classes("w-full").style(
                f"border-radius:1rem; overflow:hidden; border:1px solid {BORDER};"
            )

            with ui.column().classes("w-full rounded-2xl p-5 gap-2").style(
                f"background:{SURFACE}; border:1px solid {BORDER};"
            ):
                ui.label("It's just numbers").style(f"color:{TEXT}; font-weight:600;")
                ui.label(
                    f"soundfile decoded this file into a NumPy array of shape "
                    f"{data.shape}. The first {len(info.preview)} raw amplitude "
                    f"values (channel 1 if stereo) look like this:"
                ).style(f"color:{MUTED}; font-size:0.9rem;")
                ui.label(str(info.preview)).classes("mono").style(
                    f"color:{ACCENT}; font-size:0.85rem; background:{SURFACE_2}; "
                    f"padding:0.6rem 0.8rem; border-radius:0.6rem; word-break:break-all;"
                )

            with ui.expansion("What am I looking at? (sample rate & mono vs. stereo)", icon="school").classes(
                "w-full rounded-2xl"
            ).style(f"background:{SURFACE}; border:1px solid {BORDER}; color:{TEXT};"):
                ui.markdown(
                    f"""
- **Sample rate ({info.sample_rate:,} Hz)** — the microphone/ADC measured the
  air pressure {info.sample_rate:,} times every second. Each measurement is one
  number in the array. More samples per second → higher frequencies can be
  represented (Nyquist: max frequency ≈ sample_rate / 2).
- **Duration** = number of samples ÷ sample rate = {info.num_samples:,} ÷
  {info.sample_rate:,} = **{info.duration_sec:.3f} s**.
- **Mono vs. stereo** — mono audio is a 1D array `(num_samples,)`: one
  amplitude value per moment in time. Stereo is 2D,
  `(num_samples, 2)`: a *left* and *right* value per moment, so two ears hear
  slightly different signals. This file is **{'stereo' if info.channels > 1 else 'mono'}**.
- **Amplitude** is just "how far the speaker cone should move" at that
  instant, normalized to roughly [-1, 1]. The waveform plot above is simply
  this array plotted against time — nothing more.
"""
                ).style("font-size:0.9rem;")

    async def handle_upload(e: events.UploadEventArguments) -> None:
        suffix = Path(e.file.name).suffix or ".wav"
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        await e.file.save(dest)
        render_results(dest, e.file.name)

    with ui.column().classes("w-full items-center").style(f"padding: 2.5rem 1.5rem 4rem;"):
        with ui.column().classes("w-full gap-6").style("max-width: 960px;"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("soup_kitchen").style(f"color:{ACCENT}; font-size:2rem;")
                with ui.column().classes("gap-0"):
                    ui.label("Audio Chef").style(
                        f"color:{TEXT}; font-size:1.6rem; font-weight:800; letter-spacing:-0.02em;"
                    )
                    ui.label("Day 1 — Audio as a Signal").style(
                        f"color:{MUTED}; font-size:0.9rem;"
                    )

            with ui.row().classes("w-full items-center justify-between rounded-2xl p-4").style(
                f"background:{SURFACE}; border:1px solid {BORDER};"
            ):
                ui.label("Drop a .wav file to inspect it as raw signal data").style(
                    f"color:{MUTED};"
                )
                ui.upload(
                    on_upload=handle_upload,
                    auto_upload=True,
                    label="Choose WAV file",
                ).props('accept=".wav" flat color=cyan-9').classes("max-w-xs").style(
                    f"color:{TEXT};"
                )

            result_area = ui.column().classes("w-full gap-5")
            render_placeholder()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Audio Chef", favicon="🎧", port=8080, dark=True, reload=True)
