import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import moyal


class M19Spectra:
    def __init__(self, file_path):
        """
        Initialize the M19Spectra class with the file path to the data.
        """
        self.file_path = file_path
        self.x = None
        self.y = None

    def get_adc_spectra(self, xlim=None, ylim=None, FWHM=False, normalize_y=False):
        """
        Process and plot ADC spectra from the file.
        """
        data = np.loadtxt(self.file_path, delimiter="\t")
        self.x, self.y = data[:, 0], data[:, 1]
        if normalize_y:
            self.y = self.y / np.max(self.y)
            y_label = "Normalized Y-axis"
        else:
            y_label = "ADC counts"

        # Ask for plot title
        title_input = input("Enter the plot title (leave blank for no title): ")

        user_input = input(
            "Enter the x range for the fit as 'min,max' (leave blank to skip fit): "
        )
        fit_done = False
        fit_label = "Moyal Fit (on selected range)"
        FWHM_val = None
        fwhm_x1, fwhm_x2, half_max = None, None, None

        if user_input:
            try:
                x_min, x_max = map(float, user_input.split(","))
                mask = (self.x >= x_min) & (self.x <= x_max)
                x_fit, y_fit = self.x[mask], self.y[mask]

                if len(x_fit) == 0:
                    print("No data in the specified range. Skipping fit.")
                else:

                    def moyal_func(x, mu, sigma, amplitude):
                        return amplitude * moyal.pdf(x, mu, sigma)

                    initial_mu = x_fit[np.argmax(y_fit)]
                    initial_sigma = np.std(x_fit) / 2
                    initial_amplitude = np.max(y_fit)

                    bounds = ([x_fit.min(), 0, 0], [x_fit.max(), np.inf, np.inf])

                    popt, pcov = curve_fit(
                        moyal_func,
                        x_fit,
                        y_fit,
                        p0=[initial_mu, initial_sigma, initial_amplitude],
                        bounds=bounds,
                    )

                    mu, sigma, amplitude = popt

                    print("Moyal Fit Parameters:")
                    print(f"Mu (Location): {mu}")
                    print(f"Sigma (Scale): {sigma}")
                    print(f"Amplitude: {amplitude}")

                    if FWHM:
                        x_fine = np.linspace(min(x_fit), max(x_fit), 5000)
                        y_fine = moyal_func(x_fine, *popt)
                        half_max = np.max(y_fine) / 2
                        above_half = np.where(y_fine >= half_max)[0]
                        if len(above_half) > 1:
                            fwhm_x1 = x_fine[above_half[0]]
                            r_idx = above_half[-1]
                            if r_idx + 1 < len(x_fine):
                                x_right_l = x_fine[r_idx]
                                x_right_r = x_fine[r_idx + 1]
                                y_right_l = y_fine[r_idx]
                                y_right_r = y_fine[r_idx + 1]
                                fwhm_x2 = x_right_l + (half_max - y_right_l) * (
                                    x_right_r - x_right_l
                                ) / (y_right_r - y_right_l)
                            else:
                                fwhm_x2 = x_fine[r_idx]
                            FWHM_val = fwhm_x2 - fwhm_x1
                            print(f"FWHM: {FWHM_val}")
                            print(f"Right FWHM intersection at x = {fwhm_x2:.4f}")
                        else:
                            print("Could not determine FWHM.")

                    fit_label = (
                        f"Moyal Fit (on selected range)\n"
                        f"$\\mu$ = {mu:.2f}\n"
                        f"$\\sigma$ = {sigma:.2f}\n"
                        f"Amplitude = {amplitude:.2f}"
                    )
                    if FWHM and FWHM_val is not None:
                        fit_label += f"\nFWHM = {FWHM_val:.2f}"

                    fit_done = True
            except ValueError:
                print("Invalid input. Skipping fit.")

        plt.figure()
        plt.plot(
            self.x,
            self.y,
            label="ADC Spectra",
            linestyle=":",
            color="steelblue",
            alpha=1.0,
            linewidth=1,
        )
        if fit_done:
            plt.plot(
                x_fit,
                moyal_func(x_fit, *popt),
                color="red",
                label=fit_label,
                linewidth=1.75,
            )
            if (
                FWHM
                and FWHM_val is not None
                and fwhm_x1 is not None
                and fwhm_x2 is not None
            ):
                plt.hlines(
                    half_max,
                    fwhm_x1,
                    fwhm_x2,
                    color="green",
                    linestyle="dotted",
                    linewidth=2,
                    label="FWHM",
                )
                plt.vlines(
                    [fwhm_x1, fwhm_x2],
                    0,
                    half_max,
                    color="green",
                    linestyle="dotted",
                    linewidth=1,
                )
                plt.scatter(fwhm_x2, half_max, color="green", zorder=5)
                plt.annotate(
                    f"Edge = {fwhm_x2:.2f}",
                    (fwhm_x2, half_max),
                    textcoords="offset points",
                    xytext=(15, 5),
                    ha="left",
                    color="green",
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="green"),
                )

        plt.xlabel("Channel No.")
        plt.ylabel(y_label)
        if title_input.strip():
            plt.title(title_input)

        plt.minorticks_on()
        plt.grid(which="both", linestyle="--", linewidth=0.5)

        plt.legend(loc="best", fontsize=9)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        plt.tight_layout()
        plt.show()
        return self.x, self.y
