import cv2
import numpy as np
import pandas as pd
import os
import argparse
from textwrap import dedent

# ===========================
# ROI selector avec correction d'échelle
# ===========================

class ScaledRoiSelector:
    """
    On affiche une image éventuellement redimensionnée pour l'écran,
    mais on remappe les coordonnées des ROIs vers l'image originale.
    """

    def __init__(self, image_path, max_display_size=900):
        self.image_path = image_path
        self.img_orig = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.img_orig is None:
            raise FileNotFoundError(image_path)

        self.H, self.W = self.img_orig.shape[:2]

        # Calcul du facteur d'échelle pour l'affichage
        scale = 1.0
        max_dim = max(self.W, self.H)
        if max_dim > max_display_size:
            scale = max_display_size / max_dim

        self.scale = scale
        self.display = cv2.resize(
            self.img_orig,
            (int(self.W * scale), int(self.H * scale)),
            interpolation=cv2.INTER_AREA
        )

        self.rois_display = []  # ROIs en coords "affichage"
        self.current_rect = None
        self.drawing = False
        self.ix = 0
        self.iy = 0

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.current_rect = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            x0, y0 = min(self.ix, x), min(self.iy, y)
            x1, y1 = max(self.ix, x), max(self.iy, y)
            self.current_rect = (x0, y0, x1 - x0, y1 - y0)
            self._refresh_window()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x0, y0 = min(self.ix, x), min(self.iy, y)
            x1, y1 = max(self.ix, x), max(self.iy, y)
            w, h = x1 - x0, y1 - y0
            if w > 2 and h > 2:
                self.rois_display.append((x0, y0, w, h))
            self.current_rect = None
            self._refresh_window()

    def _refresh_window(self):
        img_show = self.display.copy()
        # Dessine les anciens ROIs
        for i, (x, y, w, h) in enumerate(self.rois_display, start=1):
            cv2.rectangle(img_show, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_show, f"#{i}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Dessine le rect en cours
        if self.current_rect is not None:
            x, y, w, h = self.current_rect
            cv2.rectangle(img_show, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Texte d'aide
        y0 = 25
        help_lines = [
            "Dessiner des ROIs avec la souris",
            "Ordre conseillé : test line (#impair), puis fond blanc (#pair)",
            "ENTER = valider, BACKSPACE = annuler dernier, ESC = tout annuler"
        ]
        for line in help_lines:
            cv2.putText(img_show, line, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y0 += 25

        cv2.imshow("ROI selector", img_show)

    def run(self):
        cv2.namedWindow("ROI selector", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("ROI selector", self._mouse_callback)
        self._refresh_window()

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 13:  # ENTER
                break
            elif key in (8, 127):  # BACKSPACE / SUPPR
                if self.rois_display:
                    self.rois_display.pop()
                    self._refresh_window()
            elif key == 27:  # ESC
                self.rois_display = []
                break

        cv2.destroyWindow("ROI selector")

        # Remap en coords originales
        rois_original = []
        for (xd, yd, wd, hd) in self.rois_display:
            x0 = int(round(xd / self.scale))
            y0 = int(round(yd / self.scale))
            w0 = int(round(wd / self.scale))
            h0 = int(round(hd / self.scale))
            # Clamp pour rester dans l'image
            x0 = max(0, min(x0, self.W - 1))
            y0 = max(0, min(y0, self.H - 1))
            w0 = max(1, min(w0, self.W - x0))
            h0 = max(1, min(h0, self.H - y0))
            rois_original.append((x0, y0, w0, h0))

        return rois_original, self.img_orig


# ===========================
# Mesure d'intensité sur ROIs
# ===========================

def measure_rois(image_path, max_display=900):
    selector = ScaledRoiSelector(image_path, max_display_size=max_display)
    rois, img = selector.run()
    if not rois:
        print(f"[INFO] Aucune ROI sélectionnée pour {image_path}.")
        return []

    # Calcul intensités
    results = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for idx, (x, y, w, h) in enumerate(rois, start=1):
        sub_gray = gray[y:y+h, x:x+w]
        sub_bgr = img[y:y+h, x:x+w]
        b, g, r = cv2.split(sub_bgr)

        res = dict(
            roi_index=idx,
            x=x, y=y, w=w, h=h,
            mean_gray=float(np.mean(sub_gray)),
            mean_R=float(np.mean(r)),
            mean_G=float(np.mean(g)),
            mean_B=float(np.mean(b))
        )
        results.append(res)

    return results


def build_dataframe_for_image(image_path, mode, max_display=900):
    """
    Mesure les ROIs pour une image donnée et construit un DataFrame
    avec rôles (test_line / background_below) et pair_id.
    """
    rows = measure_rois(image_path, max_display=max_display)
    if not rows:
        return pd.DataFrame()  # vide

    df = pd.DataFrame(rows)
    df["image"] = os.path.basename(image_path)

    # type de fond en fonction du mode
    if mode == "au":
        df["background_type"] = "black"   # fond noir pour AuNPs
    else:
        df["background_type"] = "white"   # fond blanc pour core-shells + inconnues

    # Rôle en fonction de l'index (1 = test line, 2 = background, etc.)
    roi_roles = []
    pair_ids = []
    for idx in df["roi_index"]:
        if idx % 2 == 1:
            roi_roles.append("test_line")
        else:
            roi_roles.append("background_below")
        pair_ids.append((idx + 1) // 2)  # (1,2)->1 ; (3,4)->2 ; etc.

    df["roi_role"] = roi_roles
    df["pair_id"] = pair_ids

    # Mode: au / cs / unk
    df["mode"] = mode

    return df


# ===========================
# Main CLI
# ===========================

def main():
    ap = argparse.ArgumentParser(
        description="Outil de mesure d'intensité dans des ROIs pour bandelettes (AuNPs, core-shell, inconnues)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
        Utilisation typique :

          1) Préparer des images de bandelettes (AuNPs, core-shell, inconnues)
          2) Lancer le script en précisant le mode et la liste d'images :

             - Pour les AuNPs (fond noir) :
               python roi_intensity_tool.py --mode au --images img1.jpg img2.jpg --out calib_AuNPs_intensities.xlsx

             - Pour les core-shells (fond blanc) :
               python roi_intensity_tool.py --mode cs --images img3.jpg img4.jpg --out calib_coreshell_intensities.xlsx

             - Pour les bandelettes inconnues (fond blanc) :
               python roi_intensity_tool.py --mode unk --images unknown1.jpg unknown2.jpg --out unknown_samples_intensities.xlsx

          3) Dans la fenêtre de sélection :
             - Dessiner d'abord la ROI sur la LIGNE DE TEST (#1),
             - Puis la ROI sur la ZONE BLANCHE juste en dessous (#2),
             - Puis #3 test line sur une autre bandelette, #4 fond blanc, etc.
             - ENTER pour valider, BACKSPACE pour annuler la dernière ROI, ESC pour tout annuler.

          4) Le script produit un fichier Excel avec les colonnes :
             image, roi_index, pair_id, roi_role (test_line/background_below),
             mean_gray, mean_R, mean_G, mean_B, background_type, mode.
        """)
    )

    ap.add_argument("--mode",
                    choices=["au", "cs", "unk"],
                    required=True,
                    help="Type de bandelette : 'au' = AuNPs (fond noir), 'cs' = core-shells (fond blanc), 'unk' = inconnues (fond blanc).")

    # On autorise soit une seule image, soit plusieurs
    ap.add_argument("--image", help="Chemin d'une seule image à analyser")
    ap.add_argument("--images", nargs="+", help="Liste de plusieurs images à analyser")

    ap.add_argument("--out", default=None,
                    help="Fichier Excel de sortie (.xlsx). "
                         "Par défaut: AuNPs_intensities.xlsx / CoreShells_intensities.xlsx / Unknown_intensities.xlsx")

    ap.add_argument("--max-display", type=int, default=900,
                    help="Taille max d'affichage pour l'image (plus petit si écran petit)")

    args = ap.parse_args()

    # Gestion des images
    image_paths = []
    if args.images is not None:
        image_paths.extend(args.images)
    if args.image is not None:
        image_paths.append(args.image)

    if not image_paths:
        raise ValueError("Vous devez fournir --image ou --images.")

    # Vérification des chemins
    image_paths = [p for p in image_paths if os.path.isfile(p)]
    if not image_paths:
        raise FileNotFoundError("Aucune image valide fournie.")

    # Construction du DataFrame global
    all_dfs = []
    for img_path in image_paths:
        print(f"[INFO] Traitement de l'image: {img_path}")
        df_img = build_dataframe_for_image(img_path, mode=args.mode, max_display=args.max_display)
        if not df_img.empty:
            all_dfs.append(df_img)

    if not all_dfs:
        print("[INFO] Aucune mesure enregistrée (aucune ROI sur toutes les images).")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Nom de fichier Excel par défaut si non fourni
    out_xlsx = args.out
    if out_xlsx is None:
        if args.mode == "au":
            base = "AuNPs_intensities"
        elif args.mode == "cs":
            base = "CoreShells_intensities"
        else:
            base = "Unknown_intensities"
        out_xlsx = base + ".xlsx"

    # Sauvegarde en Excel
    # Nécessite normalement 'openpyxl' installé (pip install openpyxl)
    df_all.to_excel(out_xlsx, index=False)
    print(f"[OK] Mesures sauvegardées dans {out_xlsx}")
    print(df_all)


if __name__ == "__main__":
    main()
