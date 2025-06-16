from pathlib import Path
from typing import List

import joblib
import numpy as np
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model

# ──────────────────────────────────────────────
# 설정 ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).resolve().with_name("lstm_3in_7out_model.h5")
SCALER_PATH = Path(__file__).resolve().with_name("scaler_3in_7out.save")

# ──────────────────────────────────────────────
# 초기화 -------------------------------------------------------------------------

app = Flask(__name__)

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(f"✖ 모델 로드 실패: {MODEL_PATH}") from exc

try:
    scaler = joblib.load(SCALER_PATH)
except Exception as exc:
    raise RuntimeError(f"✖ 스케일러 로드 실패: {SCALER_PATH}") from exc


# ──────────────────────────────────────────────
# 유틸리티 함수 ------------------------------------------------------------------

def validate_input(values: List[float]) -> np.ndarray:
    """수요량 3개를 받아 2D 배열(N,1)로 변환 후 반환."""
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("`demandValues`는 숫자 3개짜리 리스트여야 합니다.")
    try:
        return np.array(values, dtype=float).reshape(-1, 1)
    except Exception as exc:
        raise ValueError("`demandValues` 항목을 실수(float)로 변환할 수 없습니다.") from exc


def predict_sequence(input_scaled: np.ndarray) -> np.ndarray:
    """정규화된 입력을 받아 7일 예측(정규화 값) 반환."""
    input_seq = input_scaled.reshape(1, 3, 1)        # (batch, time, features)
    pred_scaled = model.predict(input_seq, verbose=0)[0]  # (7,)
    return np.clip(pred_scaled, 0.0, 1.0)            # 안정성 확보


def inverse_transform(pred_scaled: np.ndarray) -> np.ndarray:
    """정규화된 7-일 시퀀스를 scaler 기준으로 역변환해 정수(일 수요량) 배열 반환."""
    padded = np.column_stack([pred_scaled, np.zeros((7, scaler.n_features_in_ - 1))])
    return scaler.inverse_transform(padded)[:, 0].round().astype(int)


# ──────────────────────────────────────────────
# 라우팅 -------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True) or {}
        input_raw = payload.get("demandValues")
        input_scaled = scaler.transform(validate_input(input_raw))
        pred_scaled = predict_sequence(input_scaled)
        pred_actual = inverse_transform(pred_scaled)

        return jsonify(
            status="success",
            input=input_raw,
            predicted_daily=pred_actual.tolist(),
            predicted_total=int(pred_actual.sum())
        )

    except ValueError as ve:
        return jsonify(status="error", message=str(ve)), 400
    except Exception as e:
        return jsonify(status="error", message=str(e)), 500


# ──────────────────────────────────────────────
# 진입점 -------------------------------------------------------------------------

if __name__ == "__main__":
    # debug=True 대신 production 환경에서는 Gunicorn 등의 WSGI 서버 사용 권장
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)