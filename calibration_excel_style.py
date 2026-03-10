import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
#  CONFIG
# ==========================

AU_FILE = "calibration_AuNPs_intensities.xlsx"
CS_FILE = "calibration_CoreShells_intensities.xlsx"
UNK_FILE = "unknown_samples_intensities.xlsx"
UNK_OUT = "unknown_estimates.xlsx"

# Concentrations connues dans l'ordre des pair_id 1..6
CONCS_ALL = np.array([0.0, 0.01, 0.1, 0.5, 1.0, 2.0], dtype=float)


# ==========================
#  EXTRACTION DES RATIOS
# ==========================

def extract_ratios(df: pd.DataFrame, channel: str = "mean_R", debug: bool = True) -> pd.DataFrame:
    """
    Transforme un fichier ROI (sorti de roi_intensity_tool.py) en tableau ratios:

    - ratio = channel(test_line) / channel(background_below)
    - groupé par (image, pair_id)

    Retourne DataFrame: [image, pair_id, ratio, test_val, bg_val]
    """
    required = {"image", "pair_id", "roi_role", channel}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier: {missing}")

    rows = []
    for (img, pid), sub in df.groupby(["image", "pair_id"], sort=True):
        line = sub[sub["roi_role"] == "test_line"]
        bg = sub[sub["roi_role"] == "background_below"]

        if line.empty or bg.empty:
            if debug:
                print(f"[DEBUG] image={img} pair_id={pid} -> MISSING test/background")
            continue

        test_val = float(line[channel].iloc[0])
        bg_val = float(bg[channel].iloc[0])
        ratio = test_val / bg_val if bg_val != 0 else np.nan

        if debug:
            print(f"[DEBUG] image={img} pair_id={pid} "
                  f"{channel}_test={test_val:.3f} {channel}_bg={bg_val:.3f} ratio={ratio:.5f}")

        rows.append({
            "image": img,
            "pair_id": int(pid),
            "ratio": ratio,
            "test_val": test_val,
            "bg_val": bg_val,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out.sort_values(["image", "pair_id"], inplace=True)
    return out


# ==========================
#  CALIBRATION
# ==========================

def build_calibration(df_rat: pd.DataFrame, label: str) -> dict:
    """
    Construit une calibration linéaire:
        log10(C) = m * ratio + b

    - suppose un seul 'image' dans df_rat (Bus.jpg ou Car.jpg)
    - associe pair_id 1..6 -> CONCS_ALL
    - supprime le point C=0 (pair_id 1 si tout est aligné)
    """
    if df_rat.empty:
        raise ValueError(f"Aucun ratio trouvé pour {label}.")

    # On aligne par pair_id = 1..6
    df_one = df_rat.copy()
    df_one = df_one[df_one["pair_id"].between(1, len(CONCS_ALL))]
    df_one.sort_values("pair_id", inplace=True)

    # Ratios dans l'ordre pair_id
    ratios_all = df_one["ratio"].to_numpy(dtype=float)
    concs_all = CONCS_ALL[:len(ratios_all)]

    # On enlève C=0
    mask = concs_all > 0
    concs = concs_all[mask]
    ratios = ratios_all[mask]
    logC = np.log10(concs)

    # Régression linéaire
    m, b = np.polyfit(ratios, logC, 1)
    y_pred = m * ratios + b
    ss_res = float(np.sum((logC - y_pred) ** 2))
    ss_tot = float(np.sum((logC - np.mean(logC)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return {
        "label": label,
        "ratios": ratios,
        "logC": logC,
        "slope": float(m),
        "intercept": float(b),
        "r2": float(r2),
    }


def estimate_unknowns(df_rat_unk: pd.DataFrame, calib: dict, calib_code: str) -> pd.DataFrame:
    """
    Applique la calibration choisie aux ratios inconnus.
    """
    m = calib["slope"]
    b = calib["intercept"]

    rows = []
    for _, r in df_rat_unk.iterrows():
        ratio = float(r["ratio"])
        logC_est = m * ratio + b
        C_est = 10 ** logC_est
        rows.append({
            "image": r["image"],
            "pair_id": int(r["pair_id"]),
            "ratio": ratio,
            "logC_est": float(logC_est),
            "C_est": float(C_est),
            "calibration_used": calib_code,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values(["image", "pair_id"], inplace=True)
    return out


# ==========================
#  PLOT
# ==========================

def plot_calibration(calib: dict, unknown_df: pd.DataFrame | None = None) -> None:
    """
    Trace:
      - points calibration (bleu)
      - droite (non coupée)
      - points inconnus (rouge X) optionnels

    Sans quadrillage.
    """
    ratios = np.array(calib["ratios"], dtype=float)
    logC = np.array(calib["logC"], dtype=float)
    m = calib["slope"]
    b = calib["intercept"]
    r2 = calib["r2"]
    label = calib["label"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Points calibration
    ax.scatter(ratios, logC, s=60, label="Points de calibration")

    # Points inconnus (optionnel) - on les place avec leurs logC estimés
    all_x = list(ratios)
    if unknown_df is not None and not unknown_df.empty:
        ax.scatter(
            unknown_df["ratio"].astype(float),
            unknown_df["logC_est"].astype(float),
            s=90,
            marker="x",
            linewidths=2,
            color="red",
            label="Échantillons inconnus",
        )
        all_x += list(unknown_df["ratio"].astype(float).values)

    # Domaine de la droite = couvrir calibration + inconnus (donc jamais coupée)
    x_min = float(np.min(all_x)) - 0.01
    x_max = float(np.max(all_x)) + 0.01
    xs = np.linspace(x_min, x_max, 200)
    ys = m * xs + b
    ax.plot(xs, ys, linestyle="--", label=f"Fit linéaire (R²={r2:.4f})")

    ax.set_title(f"Calibration – {label}")
    ax.set_xlabel("Ratio intensité (test / fond)")
    ax.set_ylabel("log10(concentration)")

    ax.grid(False)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ==========================
#  MAIN
# ==========================

def main():
    # ---- Chargement et calibration AuNPs
    print("\n=== AuNPs (Bus.jpg) ===")
    df_au = pd.read_excel(AU_FILE)
    df_rat_au = extract_ratios(df_au, channel="mean_R", debug=True)
    cal_au = build_calibration(df_rat_au, "AuNPs")
    print("AuNPs:", cal_au)
    plot_calibration(cal_au)

    # ---- Chargement et calibration Core-shells
    print("\n=== Core-shells (Car.jpg) ===")
    df_cs = pd.read_excel(CS_FILE)
    df_rat_cs = extract_ratios(df_cs, channel="mean_R", debug=True)
    cal_cs = build_calibration(df_rat_cs, "Core-shells")
    print("Core-shells:", cal_cs)
    plot_calibration(cal_cs)

    # ---- Inconnues
    print("\n=== Inconnues (Plane.jpg) ===")
    df_unk = pd.read_excel(UNK_FILE)
    df_rat_unk = extract_ratios(df_unk, channel="mean_R", debug=True)

    if df_rat_unk.empty:
        print("[INFO] Aucun ratio inconnu trouvé. Fin.")
        return

    # Choix calibration pour chaque image inconnue (si plusieurs)
    all_est = []
    for img, sub in df_rat_unk.groupby("image", sort=True):
        choice = input(f"{img}: utiliser calibration AuNPs (au) ou Core-shells (cs) ? [cs] : ").strip().lower()
        if choice == "":
            choice = "cs"
        if choice not in ("au", "cs"):
            print("[WARN] choix invalide -> cs par défaut.")
            choice = "cs"

        calib = cal_au if choice == "au" else cal_cs
        est = estimate_unknowns(sub, calib, choice)
        all_est.append(est)

    df_est = pd.concat(all_est, ignore_index=True)
    df_est.to_excel(UNK_OUT, index=False)
    print(f"\n[OK] Résultats inconnus enregistrés dans {UNK_OUT}")
    print(df_est)

    # ---- Affichage : calibration utilisée + inconnus dessus
    # (si plusieurs images inconnues avec choix différents, on plotte image par image)
    for img, sub_est in df_est.groupby("image", sort=True):
        used = sub_est["calibration_used"].iloc[0]
        calib = cal_au if used == "au" else cal_cs
        print(f"\n[INFO] Plot {img} sur calibration {calib['label']} ({used})")
        plot_calibration(calib, unknown_df=sub_est)


if __name__ == "__main__":
    main()
