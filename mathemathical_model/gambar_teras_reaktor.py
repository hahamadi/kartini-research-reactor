import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ==========================================
# KOORDINAT POSISI TERAS
# Disusun agar bentuknya mirip gambar Kartini
# ==========================================
positions = [
    (185, 27), (217, 29), (245, 31), (155, 32), (272, 40),
    (128, 45), (200, 49), (172, 55), (231, 59), (296, 62),
    (104, 63), (143, 69), (259, 70), (217, 79), (183, 81),
    (85, 83), (317, 83), (283, 88), (115, 89), (239, 89),
    (157, 95), (332, 110), (70, 111), (201, 112), (271, 112),
    (295, 112), (99, 113), (131, 115), (225, 117), (172, 118),
    (61, 139), (86, 139), (117, 139), (154, 139), (251, 139),
    (310, 139), (339, 139), (280, 140), (209, 143), (184, 148),
    (256, 164), (57, 167), (172, 167), (86, 169), (314, 169),
    (343, 169), (201, 171), (284, 171), (145, 173), (229, 173),
    (113, 175), (213, 191), (185, 193), (149, 195), (250, 196),
    (59, 197), (281, 197), (85, 200), (119, 200), (341, 200),
    (313, 203), (173, 217), (233, 217), (201, 219), (101, 224),
    (266, 224), (328, 224), (69, 225), (137, 225), (293, 225),
    (158, 241), (241, 242), (283, 249), (83, 251), (115, 253),
    (213, 253), (316, 253), (187, 257), (259, 267), (140, 269),
    (104, 273), (233, 274), (297, 275), (169, 278), (200, 283),
    (127, 292), (272, 293), (155, 302), (243, 305), (184, 308),
    (214, 311)
]

# ==========================================
# STYLE WARNA
# ==========================================
STYLE = {
    "fuel": {
        "facecolor": "white",
        "edgecolor": "#666666",
        "textcolor": "#555555",
        "linewidth": 1.4
    },
    "control_rod": {
        "facecolor": "#ff7f50",
        "edgecolor": "#666666",
        "textcolor": "black",
        "linewidth": 1.4
    },
    "dummy": {
        "facecolor": "#6f79ff",
        "edgecolor": "#666666",
        "textcolor": "black",
        "linewidth": 1.4
    },
    "pneumatic": {
        "facecolor": "#6be86a",
        "edgecolor": "#666666",
        "textcolor": "black",
        "linewidth": 1.4
    },
    "neutron_source": {
        "facecolor": "#f0dd5f",
        "edgecolor": "#666666",
        "textcolor": "black",
        "linewidth": 1.4
    },
    "empty": {
        "facecolor": "#dd77dd",
        "edgecolor": "#666666",
        "textcolor": "black",
        "linewidth": 1.4
    },
    "detector_housing": {
        "facecolor": "white",
        "edgecolor": "#666666",
        "textcolor": "#555555",
        "linewidth": 1.4
    }
}

# ==========================================
# POSISI KOMPONEN KHUSUS
# ==========================================
dummy_positions = [
    (128, 45), (104, 63), (85, 83), (61, 139),
    (296, 62), (332, 110), (339, 139), (341, 200),
    (83, 251), (104, 273), (297, 275), (243, 305),
    (184, 308), (214, 311), (59, 197)
]

special_map = {
    (200, 49): ("control_rod", "E1"),
    (149, 195): ("control_rod", "C0"),
    (250, 196): ("control_rod", "C6"),
    (317, 83): ("empty", "F6"),
    (343, 169): ("pneumatic", "PN"),
    (155, 302): ("neutron_source", "Am\nBe"),
    (201, 171): ("detector_housing", "CT"),
}

for pos in dummy_positions:
    special_map[pos] = ("dummy", "G")

# override ulang supaya tidak tertimpa
special_map[(317, 83)] = ("empty", "F6")
special_map[(343, 169)] = ("pneumatic", "PN")
special_map[(155, 302)] = ("neutron_source", "Am\nBe")
special_map[(200, 49)] = ("control_rod", "E1")
special_map[(149, 195)] = ("control_rod", "C0")
special_map[(250, 196)] = ("control_rod", "C6")
special_map[(201, 171)] = ("detector_housing", "CT")

# ==========================================
# OPSI TAMPILAN
# ==========================================
FIGSIZE = (12, 7)
CORE_RADIUS = 14.5
SHOW_FUEL_LABELS = True
OUTPUT_FILE = "kartini_core_like.png"

# ==========================================
# GAMBAR
# ==========================================
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.set_facecolor("#d9d9d9")

fuel_start_number = 9530

for i, (x, y) in enumerate(positions):
    kind, label = special_map.get((x, y), ("fuel", str(fuel_start_number + i)))
    style = STYLE[kind]

    circle = Circle(
        (x, y),
        CORE_RADIUS,
        facecolor=style["facecolor"],
        edgecolor=style["edgecolor"],
        linewidth=style["linewidth"]
    )
    ax.add_patch(circle)

    # tampilkan label
    if kind == "fuel" and not SHOW_FUEL_LABELS:
        pass
    else:
        fontsize = 8 if kind == "fuel" else 10
        fontweight = "normal" if kind == "fuel" else "bold"

        ax.text(
            x, y, label,
            ha="center", va="center",
            fontsize=fontsize,
            color=style["textcolor"],
            fontweight=fontweight
        )

# ==========================================
# ANOTASI DETECTOR
# ==========================================
ax.annotate(
    "Detector",
    xy=(317, 83),
    xytext=(345, 55),
    arrowprops=dict(arrowstyle="->", lw=1.5, color="#666666"),
    fontsize=13,
    fontweight="bold"
)

# ==========================================
# LEGENDA
# ==========================================
legend_x = 410
legend_y0 = 140
legend_gap = 35

legend_items = [
    ("control_rod", "Control rods"),
    ("dummy", "Dummy"),
    ("pneumatic", "Pneumatic"),
    ("neutron_source", "Neutron source"),
    ("empty", "Empty"),
    ("fuel", "Fuel rods"),
    ("detector_housing", "Detector housing")
]

for i, (kind, text) in enumerate(legend_items):
    y = legend_y0 + i * legend_gap
    style = STYLE[kind]

    circ = Circle(
        (legend_x, y),
        14,
        facecolor=style["facecolor"],
        edgecolor=style["edgecolor"],
        linewidth=style["linewidth"]
    )
    ax.add_patch(circ)

    if kind == "dummy":
        ax.text(legend_x, y, "G", ha="center", va="center", fontsize=10, fontweight="bold")
    elif kind == "pneumatic":
        ax.text(legend_x, y, "PN", ha="center", va="center", fontsize=8, fontweight="bold")
    elif kind == "neutron_source":
        ax.text(legend_x, y, "Am\nBe", ha="center", va="center", fontsize=7, fontweight="bold")
    elif kind == "fuel":
        ax.text(legend_x, y, "9536", ha="center", va="center", fontsize=7, color="#555555")

    ax.text(
        legend_x + 25, y,
        f"-  {text}",
        va="center",
        fontsize=12,
        fontweight="bold"
    )

# ==========================================
# RAPIIKAN TAMPILAN
# ==========================================
ax.set_xlim(20, 460)
ax.set_ylim(330, 0)   # dibalik supaya mirip layout gambar asli
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()

print(f"Gambar disimpan sebagai: {OUTPUT_FILE}")