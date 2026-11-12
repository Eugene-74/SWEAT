
from PIL import Image
import numpy as np
import tifffile
import colorsys

# ==================================
# FICHIERS
# ==================================

for ENTREE in ["arriere", "avant"]:

    SORTIE = f"{ENTREE}_separee.tif"
    SORTIE_MODIFIEE = f"{ENTREE}_couleurs_modifiees.tif"
    SORTIE_APERCU_PULL = f"{ENTREE}_apercu_pull.png"

    ENTREE = f"{ENTREE}.tif"

    # ==================================
    # COULEURS A DETECTER (RGB)
    # ==================================

    COULEUR_1 = (255, 255, 255)  # Blanc
    COULEUR_2 = (255, 165, 0)    # Orange
    COULEUR_3 = (0, 0, 100)      # Bleu foncé (FOND)

    # ==================================
    # NOUVELLES COULEURS (RGB)
    # ==================================

    # NOUVELLE_COULEUR_1 = (255, 255, 255)    # Blanc ->
    # NOUVELLE_COULEUR_2 = (217, 134, 74)    # Orange ->
    # NOUVELLE_COULEUR_3 = (121 , 164 , 219 )  # Bleu foncé -> CORNFLOWER BLUE(#79A4DB)

    # NOUVELLE_COULEUR_1 = (255, 255, 255)    # Blanc ->
    # NOUVELLE_COULEUR_2 = (217, 119, 87)    # Orange ->
    # NOUVELLE_COULEUR_3 = (128, 201, 242)  # Bleu foncé -> SKY BLUE(#80c9f2)


    NOUVELLE_COULEUR_1 = (255, 255, 255)    # Blanc ->
    NOUVELLE_COULEUR_2 = (233, 138, 45)    # Orange ->
    NOUVELLE_COULEUR_3 = (4, 28, 119)  # Bleu foncé -> OXFORD NAVY(#041c77)

    # NOUVELLE_COULEUR_1 = (255, 255, 255)    # Blanc ->
    # NOUVELLE_COULEUR_2 = (230, 162, 60)    # Orange ->
    # NOUVELLE_COULEUR_3 = (4, 28, 119)  # Bleu foncé -> OXFORD NAVY(#041c77)

    # NOUVELLE_COULEUR_1 = (255, 255, 255)  # Blanc
    # NOUVELLE_COULEUR_2 = (100, 149, 200) # Bleu acier
    # NOUVELLE_COULEUR_3 = (4, 28, 119)    # Oxford Navy


    # Noms des calques
    NOM_1 = "couleur_1"
    NOM_2 = "couleur_2"
    NOM_3 = "couleur_3"

    # ==================================
    # TOLERANCES HSV
    # ==================================

    TOLERANCE_H = 30
    TOLERANCE_S = 0.25
    TOLERANCE_V = 0.25

    # ==================================
    # CHARGEMENT IMAGE
    # ==================================

    img = Image.open(ENTREE).convert("RGBA")

    pixels = np.array(img)
    pixels[:, :, 3] = 255

    hauteur, largeur, _ = pixels.shape

    # Conversion RGB vers HSV
    hsv = np.array([
        [colorsys.rgb_to_hsv(r/255, g/255, b/255)
        for r, g, b in ligne]
        for ligne in pixels[:, :, :3]
    ])

    H = hsv[:, :, 0] * 360
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # ==================================
    # DETECTION DES COULEURS
    # ==================================

    def difference_teinte(couleur_rgb):

        r, g, b = couleur_rgb

        h_ref, _, _ = colorsys.rgb_to_hsv(
            r/255, g/255, b/255
        )
        h_ref *= 360

        diff_h = np.abs(H - h_ref)
        return np.minimum(diff_h, 360 - diff_h)


    def detecter_couleur(couleur_rgb):

        r, g, b = couleur_rgb

        _, s_ref, v_ref = colorsys.rgb_to_hsv(
            r/255, g/255, b/255
        )
        diff_h = difference_teinte(couleur_rgb)

        if couleur_rgb == COULEUR_3:

            # Détection spécifique du bleu foncé
            masque = (
                (H >= 200) &
                (H <= 260) &
                (S >= 0.35) &
                (V <= 0.65)
            )

        else:

            masque = (
                (diff_h <= TOLERANCE_H) &
                (np.abs(S - s_ref) <= TOLERANCE_S) &
                (np.abs(V - v_ref) <= TOLERANCE_V)
            )

        return masque


    def ajouter_bordures_antialias(masque, couleur_rgb):

        masque_etendu = np.pad(masque, 1)
        voisins = np.zeros_like(masque)

        for decalage_y in range(3):
            for decalage_x in range(3):
                if decalage_x == 1 and decalage_y == 1:
                    continue
                voisins |= masque_etendu[
                    decalage_y:decalage_y + hauteur,
                    decalage_x:decalage_x + largeur
                ]

        if couleur_rgb == COULEUR_1:
            couleur_compatible = (S <= 0.15) & (V >= 0.35)
        elif couleur_rgb == COULEUR_3:
            couleur_compatible = (
                (H >= 180) &
                (H <= 280) &
                (S >= 0.05)
            )
        else:
            couleur_compatible = (
                (difference_teinte(couleur_rgb) <= 30) &
                (S >= 0.05) &
                (V >= 0.5)
            )

        return masque | (voisins & couleur_compatible)


    COULEURS = {
        NOM_1: (COULEUR_1, NOUVELLE_COULEUR_1),
        NOM_2: (COULEUR_2, NOUVELLE_COULEUR_2),
        NOM_3: (COULEUR_3, NOUVELLE_COULEUR_3)
    }


    def completer_masques(masques):

        noms = list(masques)
        masques_initiaux = [masques[nom].copy() for nom in noms]
        masques_cumules = np.logical_or.reduce(masques_initiaux)
        non_classes = ~masques_cumules

        if not np.any(non_classes):
            return

        rgb = pixels[:, :, :3].astype(np.int32)
        distances = np.stack([
            np.sum((rgb - np.array(COULEURS[nom][0], dtype=np.int32)) ** 2, axis=2)
            for nom in noms
        ])

        distance_minimale = np.min(distances, axis=0)
        couleurs_proches = distances == distance_minimale
        choix = np.argmin(distances, axis=0)
        egalites = non_classes & (np.sum(couleurs_proches, axis=0) > 1)

        if np.any(egalites):
            votes_voisins = np.zeros_like(distances)
            masques_etendus = [np.pad(masque, 1) for masque in masques_initiaux]

            for decalage_y in range(3):
                for decalage_x in range(3):
                    if decalage_x == 1 and decalage_y == 1:
                        continue
                    for index, masque_etendu in enumerate(masques_etendus):
                        votes_voisins[index] += masque_etendu[
                            decalage_y:decalage_y + hauteur,
                            decalage_x:decalage_x + largeur
                        ]

            votes_voisins = np.where(couleurs_proches, votes_voisins, -1)
            meilleur_vote = np.max(votes_voisins, axis=0)
            choix_voisin = np.argmax(votes_voisins, axis=0)
            departage = egalites & (meilleur_vote > 0)
            choix[departage] = choix_voisin[departage]

        for index, nom in enumerate(noms):
            masques[nom] |= non_classes & (choix == index)


    masques = {}

    # ==================================
    # CREATION IMAGE AVEC COULEURS CHANGEES
    # ==================================

    image_modifiee = pixels.copy()

    for nom, (ancienne, nouvelle) in COULEURS.items():

        masque = detecter_couleur(ancienne)
        masque = ajouter_bordures_antialias(masque, ancienne)
        masques[nom] = masque

    masque_proche_du_blanc = (
        (S <= TOLERANCE_S) &
        (V >= 1 - TOLERANCE_V)
    )
    masques[NOM_1] |= masque_proche_du_blanc
    masques[NOM_2] &= ~masque_proche_du_blanc
    masques[NOM_3] &= ~masque_proche_du_blanc

    completer_masques(masques)

    for nom, (_, nouvelle) in COULEURS.items():
        masque = masques[nom]
        image_modifiee[masque, :3] = nouvelle
        print(nom, ":", np.sum(masque), "pixels affectés")

    # ==================================
    # SAUVEGARDE IMAGE COMPLETE MODIFIEE
    # ==================================

    Image.fromarray(image_modifiee).save(
        SORTIE_MODIFIEE,
        format="TIFF"
    )

    # ==================================
    # APERCU DU PULL AVEC LA BRODERIE
    # ==================================

    yy, xx = np.indices((hauteur, largeur), dtype=np.float32)
    trame = (
        0.014 * np.sin(2 * np.pi * (xx + 0.7 * yy) / 7)
        + 0.008 * np.sin(2 * np.pi * (xx - 0.6 * yy) / 4)
    )
    plis = (
        0.035 * np.cos(2 * np.pi * (xx / largeur * 1.2 - yy / hauteur * 0.35))
        + 0.018 * np.cos(2 * np.pi * (yy / hauteur * 1.1 + xx / largeur * 0.2))
    )
    lumiere_tissu = np.clip(0.97 + plis + trame, 0.88, 1.06)

    apercu_pull = np.empty_like(pixels)
    couleur_tissu = np.array(NOUVELLE_COULEUR_3, dtype=np.float32)
    apercu_pull[:, :, :3] = np.clip(
        couleur_tissu[None, None, :] * lumiere_tissu[:, :, None],
        0,
        255
    ).astype(np.uint8)
    apercu_pull[:, :, 3] = 255

    trame_fil = (
        0.91
        + 0.085 * (
            0.5 + 0.5 * np.cos(2 * np.pi * (xx + 0.65 * yy) / 7)
        )
    )
    for nom in (NOM_1, NOM_2):
        _, couleur_broderie = COULEURS[nom]
        if nom == NOM_1:
            couleur_fil = np.array((240, 236, 220), dtype=np.float32)
        else:
            couleur_fil = (
                np.array(couleur_broderie, dtype=np.float32) * 0.86
                + np.array((112, 96, 88), dtype=np.float32) * 0.14
            )
        couleur_broderie_texturee = np.clip(
            couleur_fil[None, None, :]
            * lumiere_tissu[:, :, None]
            * trame_fil[:, :, None],
            0,
            255
        ).astype(np.uint8)
        apercu_pull[masques[nom], :3] = couleur_broderie_texturee[
            masques[nom]
        ]

    Image.fromarray(apercu_pull).save(
        SORTIE_APERCU_PULL,
        format="PNG"
    )

    # ==================================
    # SAUVEGARDE DES 3 CALQUES SEPARES
    # ==================================

    with tifffile.TiffWriter(SORTIE) as tif:

        for nom, masque in masques.items():

            calque = np.zeros(
                (hauteur, largeur, 4),
                dtype=np.uint8
            )

            _, nouvelle = COULEURS[nom]
            calque[masque, :3] = nouvelle
            calque[masque, 3] = 255

            tif.write(
                calque,
                photometric="rgb",
                metadata=None,
                description=nom
            )

    print("--------------------------------")
    print("Séparation terminée :", SORTIE)
    print("Image recolorée :", SORTIE_MODIFIEE)
    print("Aperçu du pull :", SORTIE_APERCU_PULL)
    print("--------------------------------")
