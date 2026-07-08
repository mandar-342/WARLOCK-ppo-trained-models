from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class ReportGenerator:
    """
    Generates a professional PDF evaluation report
    for a WARLOCK experiment.
    """

    def __init__(
        self,
        evaluation_directory: str | Path,
        plots_directory: str | Path,
    ) -> None:

        self._evaluation_dir = Path(
            evaluation_directory
        )

        self._plots_dir = Path(
            plots_directory
        )

        self._metrics_file = (
            self._evaluation_dir
            / "metrics.json"
        )

        self._report_file = (
            self._evaluation_dir
            / "report.pdf"
        )

        self._styles = getSampleStyleSheet()

        self._title_style = self._styles["Heading1"]
        self._title_style.alignment = TA_CENTER

        self._heading_style = self._styles["Heading2"]

        self._body_style = self._styles["BodyText"]

    def _load_metrics(self) -> dict:

        with self._metrics_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    @staticmethod
    def _metric_table(
        metrics: dict,
    ) -> Table:

        rows = [["Metric", "Value"]]

        for key, value in metrics.items():

            if isinstance(value, float):
                value = f"{value:.6f}"

            rows.append(
                [
                    key.replace("_", " ").title(),
                    str(value),
                ]
            )

        table = Table(
            rows,
            colWidths=[3.5 * inch, 2.0 * inch],
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.darkblue,
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "BACKGROUND",
                        (0, 1),
                        (-1, -1),
                        colors.whitesmoke,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, 0),
                        10,
                    ),
                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER",
                    ),
                ]
            )
        )

        return table
    
    def _add_title(
        self,
        story: list,
    ) -> None:

        story.append(
            Paragraph(
                "WARLOCK Evaluation Report",
                self._title_style,
            )
        )

        story.append(
            Spacer(
                1,
                0.30 * inch,
            )
        )

    def _add_summary(
        self,
        story: list,
        metrics: dict,
    ) -> None:

        story.append(
            Paragraph(
                "Portfolio Summary",
                self._heading_style,
            )
        )

        summary = [
            ["Metric", "Value"],
            [
                "Final Capital",
                f"{metrics['final_capital']:.2f}",
            ],
            [
                "Peak Capital",
                f"{metrics['peak_capital']:.2f}",
            ],
            [
                "Minimum Capital",
                f"{metrics['minimum_capital']:.2f}",
            ],
            [
                "Total Return",
                f"{metrics['total_return']:.2%}",
            ],
            [
                "Maximum Drawdown",
                f"{metrics['max_drawdown']:.2%}",
            ],
            [
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.3f}",
            ],
        ]

        table = Table(
            summary,
            colWidths=[3.5 * inch, 2.0 * inch],
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.darkgreen,
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "BACKGROUND",
                        (0, 1),
                        (-1, -1),
                        colors.beige,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, 0),
                        10,
                    ),
                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER",
                    ),
                ]
            )
        )

        story.append(table)

        story.append(
            Spacer(
                1,
                0.30 * inch,
            )
        )

    def _add_trade_summary(
        self,
        story: list,
        metrics: dict,
    ) -> None:

        story.append(
            Paragraph(
                "Trading Summary",
                self._heading_style,
            )
        )

        summary = [
            ["Metric", "Value"],
            [
                "Number of Trades",
                str(metrics["number_of_trades"]),
            ],
            [
                "Win Rate",
                f"{metrics['win_rate']:.2%}",
            ],
            [
                "Average Win",
                f"{metrics['average_win']:.4f}",
            ],
            [
                "Average Loss",
                f"{metrics['average_loss']:.4f}",
            ],
            [
                "Profit Factor",
                f"{metrics['profit_factor']:.3f}",
            ],
            [
                "Expectancy",
                f"{metrics['expectancy']:.4f}",
            ],
        ]

        table = Table(
            summary,
            colWidths=[3.5 * inch, 2.0 * inch],
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.darkred,
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "BACKGROUND",
                        (0, 1),
                        (-1, -1),
                        colors.whitesmoke,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, 0),
                        10,
                    ),
                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER",
                    ),
                ]
            )
        )

        story.append(table)

        story.append(
            Spacer(
                1,
                0.30 * inch,
            )
        )
        
        
    def _add_plots(
        self,
        story: list,
    ) -> None:
        """
        Add generated evaluation plots to the report.
        """

        story.append(
            Paragraph(
                "Performance Plots",
                self._heading_style,
            )
        )

        plot_files = [
            "equity_curve.png",
            "drawdown.png",
            "returns_distribution.png",
            "rolling_sharpe.png",
        ]

        for plot in plot_files:

            plot_path = self._plots_dir / plot

            if not plot_path.exists():
                continue

            story.append(
                Image(
                    str(plot_path),
                    width=6.8 * inch,
                    height=3.8 * inch,
                )
            )

            story.append(
                Spacer(
                    1,
                    0.20 * inch,
                )
            )

    def generate(self) -> None:
        """
        Generate the complete evaluation PDF report.
        """

        metrics = self._load_metrics()

        document = SimpleDocTemplate(
            str(self._report_file),
        )

        story = []

        self._add_title(story)

        self._add_summary(
            story,
            metrics,
        )

        self._add_trade_summary(
            story,
            metrics,
        )

        story.append(
            Paragraph(
                "Complete Performance Metrics",
                self._heading_style,
            )
        )

        story.append(
            self._metric_table(metrics)
        )

        story.append(
            Spacer(
                1,
                0.30 * inch,
            )
        )

        self._add_plots(story)

        document.build(story)