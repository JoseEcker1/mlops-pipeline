import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopy.distance


def predict(model: KMeans, scaler: StandardScaler, lat: float, lng: float) -> dict:
    
    point_scaled = scaler.transform([[lat, lng]])
    cluster = int(model.predict(point_scaled)[0])
    centroid_scaled = model.cluster_centers_[cluster]
    centroid_original = scaler.inverse_transform([centroid_scaled])[0]
    distance_km = geopy.distance.geodesic(
        (lat, lng),
        (centroid_original[0], centroid_original[1])
    ).km

    prediction = {
        "cluster": cluster,
        "distance_km": round(distance_km, 4)
    }

    return prediction