"""EDF(수면다원검사 표준 파일) 읽기 — 추가 설치 없이

    from brux.edf import read_edf
    chans = read_edf("brux1.edf")          # {채널이름: (신호 µV, 샘플링 Hz)}
"""
import numpy as np

_UNIT = {"uv": 1.0, "µv": 1.0, "mv": 1e3, "v": 1e6}


def read_header(f):
    h = f.read(256)
    n_sig = int(h[252:256])
    n_rec = int(h[236:244])
    rec_dur = float(h[244:252])
    hdr_bytes = int(h[184:192])

    def field(w):
        return [f.read(w).decode("latin-1").strip() for _ in range(n_sig)]

    labels = field(16)
    field(80)
    units = field(8)
    pmin = [float(v) for v in field(8)]
    pmax = [float(v) for v in field(8)]
    dmin = [float(v) for v in field(8)]
    dmax = [float(v) for v in field(8)]
    field(80)
    nsamp = [int(v) for v in field(8)]
    f.seek(hdr_bytes)
    return dict(n_sig=n_sig, n_rec=n_rec, rec_dur=rec_dur, labels=labels, units=units,
                pmin=pmin, pmax=pmax, dmin=dmin, dmax=dmax, nsamp=nsamp, hdr_bytes=hdr_bytes)


def read_edf(path, pick=None):
    """pick: 읽을 채널 이름 목록 (None = 전부). 단위는 µV로 맞춤"""
    with open(path, "rb") as f:
        h = read_header(f)
        rec_len = sum(h["nsamp"])
        raw = np.fromfile(f, dtype="<i2")
    n_rec = len(raw) // rec_len if h["n_rec"] < 0 else min(h["n_rec"], len(raw) // rec_len)
    raw = raw[: n_rec * rec_len].reshape(n_rec, rec_len)
    out, off = {}, 0
    for i, lab in enumerate(h["labels"]):
        n = h["nsamp"][i]
        if pick is None or lab in pick:
            d = raw[:, off:off + n].reshape(-1).astype(np.float64)
            gain = (h["pmax"][i] - h["pmin"][i]) / max(h["dmax"][i] - h["dmin"][i], 1)
            x = (d - h["dmin"][i]) * gain + h["pmin"][i]
            x *= _UNIT.get(h["units"][i].lower(), 1.0)
            out[lab] = (x.astype(np.float32), n / h["rec_dur"])
        off += n
    return out


def channel_list(path):
    with open(path, "rb") as f:
        h = read_header(f)
    return [(lab, h["nsamp"][i] / h["rec_dur"], h["units"][i]) for i, lab in enumerate(h["labels"])]


def write_edf(path, chans, rec_dur=1.0):
    """시험용: {이름: (µV 신호, Hz)} → EDF"""
    names = list(chans)
    ns = [int(chans[k][1] * rec_dur) for k in names]
    n_rec = min(len(chans[k][0]) // n for k, n in zip(names, ns))
    pad = lambda s, w: s[:w].ljust(w).encode("latin-1")
    hdr = pad("0", 8) + pad("X", 80) + pad("X", 80) + pad("01.01.26", 8) + pad("00.00.00", 8)
    hdr += pad(str(256 * (len(names) + 1)), 8) + pad("", 44) + pad(str(n_rec), 8) + pad(str(rec_dur), 8) + pad(str(len(names)), 4)
    rng = [float(np.abs(chans[k][0]).max() or 1) for k in names]
    hdr += b"".join(pad(k, 16) for k in names) + b"".join(pad("", 80) for _ in names)
    hdr += b"".join(pad("uV", 8) for _ in names)
    hdr += b"".join(pad(f"{-r:.1f}", 8) for r in rng) + b"".join(pad(f"{r:.1f}", 8) for r in rng)
    hdr += b"".join(pad("-32768", 8) for _ in names) + b"".join(pad("32767", 8) for _ in names)
    hdr += b"".join(pad("", 80) for _ in names) + b"".join(pad(str(n), 8) for n in ns) + b"".join(pad("", 32) for _ in names)
    recs = []
    for r in range(n_rec):
        for k, n, rg in zip(names, ns, rng):
            seg = chans[k][0][r * n:(r + 1) * n]
            recs.append(np.round((seg + rg) / (2 * rg) * 65535 - 32768).clip(-32768, 32767).astype("<i2").tobytes())
    with open(path, "wb") as f:
        f.write(hdr + b"".join(recs))
