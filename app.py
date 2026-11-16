import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

st.set_page_config(page_title="Simulasi Beban Cloud Hosting", layout="wide")

st.title("Pemodelan Dinamika Indeks Beban Komputasi Cloud Hosting")
st.caption("Simulasi pertumbuhan terbatas (constrained growth) beban klaster server multi-tenant terhadap promosi dan pembersihan beban")

# ---------------------------------------------------------------
# 1. Parameter model
# ---------------------------------------------------------------
st.sidebar.header("Parameter Model")
P0 = st.sidebar.slider("Beban Awal P0 (%)", 5.0, 30.0, 10.0, 1.0)
r_base = st.sidebar.slider("Laju Pertumbuhan Alami (r)", 0.05, 1.0, 0.25, 0.05)
M = st.sidebar.slider("Carrying Capacity M (%)", 50.0, 150.0, 100.0, 10.0)
t_max = st.sidebar.slider("Durasi Waktu Simulasi (Bulan)", 12, 36, 24)
dt = 0.05

st.sidebar.header("Skenario Intervensi")
pakai_promo = st.sidebar.checkbox("Aktifkan Promo Full Upgrade (Bulan 4-7)", value=True)
faktor_promo = st.sidebar.slider("Faktor kenaikan r saat promo", 1.0, 3.0, 2.0, 0.5)
pakai_optimasi = st.sidebar.checkbox("Aktifkan Pembersihan Beban (Bulan 12)", value=True)
pangkas = st.sidebar.slider("Beban dipangkas saat pembersihan (%)", 5, 30, 15, 5)

AMBANG = 80.0  # ambang peringatan arsitek (% dari kapasitas 100%)

# Waktu berbasis indeks integer (mencegah akumulasi galat floating-point)
steps = int(round(t_max / dt))
time_steps = np.array([i * dt for i in range(steps + 1)])


def simulasi(p0, r, m, promo=False, optimasi=False):
    """Metode Euler untuk dP/dt = r * P * (1 - P/M)."""
    hasil = []
    p = p0
    optim_done = False
    for t in time_steps:
        hasil.append(p)
        curr_r = r * faktor_promo if (promo and 4.0 <= t < 7.0) else r
        if optimasi and t >= 12.0 and not optim_done:
            p *= (1.0 - pangkas / 100.0)  # pangkas relatif terhadap beban saat itu
            optim_done = True
        p += curr_r * p * (1.0 - p / m) * dt
    return np.array(hasil)


def solusi_analitis(p0, r, m):
    return m / (1.0 + ((m - p0) / p0) * np.exp(-r * time_steps))


def waktu_pertama(arr, batas):
    idx = np.argmax(arr >= batas)
    return time_steps[idx] if arr[idx] >= batas else None


def fmt_bulan(t):
    return f"Bulan ke-{t:.2f}" if t is not None else "Tidak tercapai"


# ---------------------------------------------------------------
# Simulasi
# ---------------------------------------------------------------
p_normal = simulasi(P0, r_base, M)
p_intervensi = simulasi(P0, r_base, M, pakai_promo, pakai_optimasi)
p_analitis = solusi_analitis(P0, r_base, M)

p_infleksi = M / 2.0
t_infleksi = np.log((M - P0) / P0) / r_base if M > P0 else 0.0
t_infleksi_int = waktu_pertama(p_intervensi, p_infleksi)
mae = np.mean(np.abs(p_normal - p_analitis))

tab1, tab2, tab3, tab4 = st.tabs(["Normal vs Intervensi",
                                  "Sensitivitas Kapasitas M",
                                  "Verifikasi Galat",
                                  "Validasi Data Nyata"])

# ---------------------------------------------------------------
# Tab 1: Normal vs Intervensi
# ---------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Carrying Capacity (M)", f"{M:.0f} %")
    c2.metric("Infleksi Normal (P = M/2)", f"Bulan ke-{t_infleksi:.2f}")
    c3.metric("Infleksi Intervensi", fmt_bulan(t_infleksi_int))
    c4.metric(f"Beban Akhir Intervensi (t={t_max})", f"{p_intervensi[-1]:.1f} %")

    d1, d2, d3 = st.columns(3)
    d1.metric("Beban Bulan ke-7 (Normal → Intervensi)",
              f"{np.interp(7, time_steps, p_normal):.1f} % → {np.interp(7, time_steps, p_intervensi):.1f} %")
    d2.metric("Melewati Ambang 80% (Normal)", fmt_bulan(waktu_pertama(p_normal, AMBANG)))
    d3.metric("Melewati Ambang 80% (Intervensi)", fmt_bulan(waktu_pertama(p_intervensi, AMBANG)))

    fig, ax = plt.subplots(figsize=(10, 5))
    if pakai_promo:
        ax.axvspan(4, 7, color="orange", alpha=0.12, label="Periode Promo")
    ax.plot(time_steps, p_normal, label="Skenario Normal", color="#1f77b4", linewidth=2)
    ax.plot(time_steps, p_intervensi, label="Skenario Intervensi", color="#ff7f0e", linewidth=2.5)
    ax.axhline(y=M, color="red", linestyle=":", label=f"Carrying Capacity M ({M:.0f}%)")
    ax.axhline(y=AMBANG, color="purple", linestyle="-.", alpha=0.7, label="Ambang Peringatan (80%)")
    ax.axhline(y=p_infleksi, color="gray", linestyle=":", label=f"Titik Infleksi ({p_infleksi:.0f}%)")
    ax.set_title("Dinamika Utilisasi Indeks Beban Komputasi Cloud Hosting", fontsize=13, fontweight="bold")
    ax.set_xlabel("Waktu t (Bulan)")
    ax.set_ylabel("Indeks Beban Komputasi Gabungan (%)")
    ax.set_ylim(0, max(115, M + 15))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    st.pyplot(fig)

# ---------------------------------------------------------------
# Tab 2: Sensitivitas M
# ---------------------------------------------------------------
with tab2:
    nilai_M = [60.0, 100.0, 150.0]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    baris = []
    for m_uji, warna in zip(nilai_M, ["#ff7f0e", "#1f77b4", "#7f7f7f"]):
        kurva = simulasi(P0, r_base, m_uji)
        t_inf = np.log((m_uji - P0) / P0) / r_base
        ax2.plot(time_steps, kurva, color=warna, linewidth=2, label=f"M = {m_uji:.0f}%")
        ax2.plot(t_inf, m_uji / 2, "o", color=warna, markersize=8, markerfacecolor="white")
        baris.append({"M (%)": f"{m_uji:.0f}",
                      "Infleksi (%)": f"{m_uji / 2:.0f}",
                      "Waktu infleksi (bulan)": f"{t_inf:.2f}",
                      "95% M tercapai": fmt_bulan(waktu_pertama(kurva, 0.95 * m_uji))})
    ax2.set_title("Pengaruh Carrying Capacity M terhadap Kurva Logistik", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Waktu t (Bulan)")
    ax2.set_ylabel("Indeks Beban (%)")
    ax2.set_ylim(0, 160)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")
    st.pyplot(fig2)
    st.table(baris)

# ---------------------------------------------------------------
# Tab 3: Verifikasi Euler vs Analitis
# ---------------------------------------------------------------
with tab3:
    st.metric("MAE Euler vs Analitis (Skenario Normal)", f"{mae:.4f} %")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(time_steps, p_analitis, color="black", linewidth=2, label="Solusi Analitis")
    ax3.plot(time_steps, p_normal, color="#1f77b4", linestyle="--", linewidth=2, label=f"Euler (dt = {dt})")
    ax3.set_xlabel("Waktu t (Bulan)")
    ax3.set_ylabel("Indeks Beban (%)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower right")
    st.pyplot(fig3)

# ---------------------------------------------------------------
# Tab 4: Validasi dengan data monitoring nyata
# ---------------------------------------------------------------
with tab4:
    st.markdown("Rata-rata trafik masuk bulanan perangkat klaster **Cloud Star Hosting** (data center "
                "Jakarta), Desember 2024 – November 2026, kapasitas link 10 Gbps. "
                "Angka dibaca dari grafik monitoring Observium 3 tahun.")
    BULAN = ["Des 24", "Jan 25", "Feb 25", "Mar 25", "Apr 25", "Mei 25", "Jun 25", "Jul 25",
             "Agu 25", "Sep 25", "Okt 25", "Nov 25", "Des 25", "Jan 26", "Feb 26", "Mar 26",
             "Apr 26", "Mei 26", "Jun 26", "Jul 26", "Agu 26", "Sep 26", "Okt 26", "Nov 26"]
    TRAFIK = np.array([0.688, 0.687, 1.293, 1.729, 2.600, 2.911, 2.942, 2.705, 1.721, 1.597, 1.619, 1.666,
                       1.869, 1.899, 1.998, 2.233, 2.099, 2.181, 2.282, 2.268, 2.279, 2.367, 2.515, 2.556])
    PROMO = np.array([4, 5, 6, 7])  # April–Juli 2025: promo upgrade gratis
    KAPASITAS = 10.0

    unggah = st.file_uploader("Opsional: unggah CSV sendiri (kolom: bulan, trafik_gbps)", type="csv")
    if unggah is not None:
        data = np.genfromtxt(unggah, delimiter=",", names=True, dtype=None, encoding="utf-8")
        TRAFIK = np.asarray(data["trafik_gbps"], dtype=float)
        BULAN = [str(b) for b in data["bulan"]]
        PROMO = np.array([], dtype=int)
    t_data = np.arange(len(TRAFIK))
    kecualikan = st.checkbox("Keluarkan bulan promo (Apr–Jul 2025) dari fitting", value=True)
    pakai = np.ones(len(TRAFIK), bool)
    if kecualikan and len(PROMO):
        pakai[PROMO] = False

    def logistik(t, M_, r_, P0_):
        return M_ / (1 + ((M_ - P0_) / P0_) * np.exp(-r_ * t))

    def eksponensial(t, r_, P0_):
        return P0_ * np.exp(r_ * t)

    try:
        (M_f, r_f, P0_f), kov = curve_fit(logistik, t_data[pakai], TRAFIK[pakai],
                                          p0=[TRAFIK.max() * 1.2, 0.2, TRAFIK[0]], maxfev=20000)
        (r_e, P0_e), _ = curve_fit(eksponensial, t_data[pakai], TRAFIK[pakai], p0=[0.05, TRAFIK[0]], maxfev=20000)
    except RuntimeError:
        st.error("Fitting tidak konvergen untuk data ini.")
        st.stop()
    galat_M, galat_r = np.sqrt(np.diag(kov))[:2]
    mae_log = np.mean(np.abs(logistik(t_data[pakai], M_f, r_f, P0_f) - TRAFIK[pakai]))
    mae_exp = np.mean(np.abs(eksponensial(t_data[pakai], r_e, P0_e) - TRAFIK[pakai]))
    t_inf_f = np.log((M_f - P0_f) / P0_f) / r_f

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("M hasil fitting", f"{M_f:.2f} Gbps", f"± {galat_M:.2f} · {M_f / KAPASITAS * 100:.1f}% kapasitas", delta_color="off")
    e2.metric("Laju r hasil fitting", f"{r_f:.3f} / bulan", f"± {galat_r:.3f}", delta_color="off")
    e3.metric("Titik infleksi", f"Bulan ke-{t_inf_f:.2f}")
    e4.metric("MAE logistik vs eksponensial", f"{mae_log:.3f} vs {mae_exp:.3f} Gbps")
    if kecualikan and len(PROMO):
        rasio = np.mean(TRAFIK[PROMO] / logistik(PROMO, M_f, r_f, P0_f))
        st.info(f"Selama promo upgrade gratis, trafik nyata rata-rata {rasio:.1f}× dari prediksi model tanpa promo, "
                "lalu turun kembali setelah promo berakhir.")

    tt = np.linspace(0, len(TRAFIK) - 1, 300)
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    if len(PROMO):
        ax4.axvspan(PROMO.min() - 0.5, PROMO.max() + 0.5, color="orange", alpha=0.12, label="Periode promo")
    ax4.plot(tt, logistik(tt, M_f, r_f, P0_f), color="#1f3864", linewidth=2.4, label="Model logistik (fitting)")
    ax4.plot(tt, eksponensial(tt, r_e, P0_e), color="gray", linestyle="--", label="Model eksponensial (pembanding)")
    ax4.axhline(M_f, color="red", linestyle=":", label=f"M = {M_f:.2f} Gbps")
    ax4.scatter(t_data[pakai], TRAFIK[pakai], color="#1f77b4", zorder=3, label="Data nyata")
    if not pakai.all():
        ax4.scatter(t_data[~pakai], TRAFIK[~pakai], color="#ff7f0e", marker="D", zorder=3, label="Data nyata saat promo")
    ax4.set_xticks(t_data[::3])
    ax4.set_xticklabels(BULAN[::3])
    ax4.set_xlabel("Bulan")
    ax4.set_ylabel("Trafik masuk (Gbps)")
    ax4.set_ylim(0, max(TRAFIK.max(), M_f) * 1.3)
    ax4.set_title("Data Monitoring Nyata vs Model Logistik", fontsize=13, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper left", ncol=2, fontsize=9)
    st.pyplot(fig4)
