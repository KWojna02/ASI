# Predictive Maintenance System - Implementation Guide

## Quick Start Guide

This guide provides step-by-step instructions to implement the predictive maintenance system architecture.

---

## Prerequisites

### Software Requirements
- Python 3.9+
- Docker & Docker Compose
- Apache Kafka 3.x
- Apache Spark 3.x
- InfluxDB 2.x
- PostgreSQL 14+
- TensorFlow 2.x or PyTorch 2.x

### Hardware Recommendations (for simulation)
- CPU: 8+ cores
- RAM: 32GB+
- Storage: 500GB+ SSD
- GPU: Optional, for faster model training

---

## Phase 1: Infrastructure Setup (Week 1)

### Step 1: Set Up Docker Environment

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  # Zookeeper for Kafka
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"
    volumes:
      - zookeeper_data:/var/lib/zookeeper/data

  # Kafka Broker
  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
      - "9094:9094"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092,PLAINTEXT_HOST://localhost:9094
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    volumes:
      - kafka_data:/var/lib/kafka/data

  # Schema Registry
  schema-registry:
    image: confluentinc/cp-schema-registry:7.5.0
    depends_on:
      - kafka
    ports:
      - "8081:8081"
    environment:
      SCHEMA_REGISTRY_HOST_NAME: schema-registry
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: kafka:9092

  # InfluxDB
  influxdb:
    image: influxdb:2.7
    ports:
      - "8086:8086"
    environment:
      DOCKER_INFLUXDB_INIT_MODE: setup
      DOCKER_INFLUXDB_INIT_USERNAME: admin
      DOCKER_INFLUXDB_INIT_PASSWORD: adminpassword
      DOCKER_INFLUXDB_INIT_ORG: predictive-maintenance
      DOCKER_INFLUXDB_INIT_BUCKET: sensors
      DOCKER_INFLUXDB_INIT_ADMIN_TOKEN: my-super-secret-token
    volumes:
      - influxdb_data:/var/lib/influxdb2

  # PostgreSQL
  postgres:
    image: postgres:14
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: maintenance_db
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: adminpassword
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Grafana
  grafana:
    image: grafana/grafana:10.2.0
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - influxdb
      - postgres

  # MinIO (S3-compatible storage)
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: miniopassword
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

volumes:
  zookeeper_data:
  kafka_data:
  influxdb_data:
  postgres_data:
  grafana_data:
  minio_data:
```

**Launch infrastructure:**
```bash
docker-compose up -d
```

### Step 2: Create Kafka Topics

```bash
# Create topics for different data streams
docker exec -it kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic sensor-raw \
  --partitions 10 \
  --replication-factor 1

docker exec -it kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic sensor-enriched \
  --partitions 10 \
  --replication-factor 1

docker exec -it kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic predictions \
  --partitions 5 \
  --replication-factor 1

docker exec -it kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic alerts \
  --partitions 3 \
  --replication-factor 1
```

### Step 3: Initialize Database Schemas

**PostgreSQL schema** (`schema.sql`):
```sql
-- Machines table
CREATE TABLE machines (
    machine_id VARCHAR(50) PRIMARY KEY,
    machine_type VARCHAR(100) NOT NULL,
    location VARCHAR(200) NOT NULL,
    installation_date TIMESTAMP,
    last_maintenance TIMESTAMP,
    status VARCHAR(50) DEFAULT 'operational',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Maintenance history
CREATE TABLE maintenance_history (
    id SERIAL PRIMARY KEY,
    machine_id VARCHAR(50) REFERENCES machines(machine_id),
    maintenance_type VARCHAR(100),
    scheduled_date TIMESTAMP,
    completed_date TIMESTAMP,
    description TEXT,
    cost DECIMAL(10, 2),
    performed_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Predictions table
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    machine_id VARCHAR(50) REFERENCES machines(machine_id),
    timestamp TIMESTAMP NOT NULL,
    failure_probability DECIMAL(5, 4),
    time_to_failure_hours DECIMAL(10, 2),
    risk_level VARCHAR(20),
    model_version VARCHAR(50),
    features JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_machine_timestamp (machine_id, timestamp),
    INDEX idx_risk_level (risk_level)
);

-- Alerts table
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    machine_id VARCHAR(50) REFERENCES machines(machine_id),
    alert_type VARCHAR(50),
    severity VARCHAR(20),
    message TEXT,
    triggered_at TIMESTAMP NOT NULL,
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,
    acknowledged_by VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_machine_status (machine_id, status)
);

-- Model registry
CREATE TABLE model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50),
    framework VARCHAR(50),
    metrics JSONB,
    hyperparameters JSONB,
    training_date TIMESTAMP,
    deployment_date TIMESTAMP,
    status VARCHAR(50) DEFAULT 'staging',
    artifact_path TEXT,
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(model_name, model_version)
);
```

Apply schema:
```bash
docker exec -i postgres psql -U admin -d maintenance_db < schema.sql
```

---

## Phase 2: Data Ingestion Pipeline (Week 2)

### Step 1: Kafka Producer - Sensor Simulator

```python
# kafka_producer.py
from kafka import KafkaProducer
import json
import time
from sensor_data_simulator import MultiMachineSimulator

def create_producer():
    return KafkaProducer(
        bootstrap_servers=['localhost:9094'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        compression_type='gzip'
    )

def stream_sensor_data():
    producer = create_producer()
    simulator = MultiMachineSimulator(num_machines=10)
    
    print("Starting sensor data streaming...")
    
    # Generate datasets
    datasets = simulator.generate_all_machines(
        duration_hours=720,  # 30 days
        sampling_rate_hz=0.1  # 10-second intervals
    )
    
    # Convert to Kafka messages
    messages = simulator.export_to_kafka_format(datasets)
    
    # Stream messages with realistic timing
    for i, message in enumerate(messages):
        producer.send('sensor-raw', value=message)
        
        if i % 100 == 0:
            print(f"Sent {i} messages...")
            producer.flush()
        
        # Simulate real-time data (10-second intervals)
        time.sleep(0.01)  # Adjust for faster simulation
    
    producer.flush()
    print(f"Streaming complete. Sent {len(messages)} messages.")

if __name__ == '__main__':
    stream_sensor_data()
```

Run the producer:
```bash
python kafka_producer.py
```

### Step 2: Kafka Consumer - InfluxDB Writer

```python
# kafka_to_influxdb.py
from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json

# InfluxDB configuration
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "my-super-secret-token"
INFLUXDB_ORG = "predictive-maintenance"
INFLUXDB_BUCKET = "sensors"

def create_consumer():
    return KafkaConsumer(
        'sensor-raw',
        bootstrap_servers=['localhost:9094'],
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id='influxdb-writer',
        auto_offset_reset='earliest'
    )

def write_to_influxdb():
    consumer = create_consumer()
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    
    print("Starting InfluxDB writer...")
    
    for message in consumer:
        data = message.value
        
        # Create InfluxDB point
        point = Point("sensor_readings") \
            .tag("machine_id", data['machine_id']) \
            .tag("machine_type", data['metadata']['machine_type']) \
            .tag("location", data['metadata']['location']) \
            .field("vibration_x", data['sensors']['vibration_x']) \
            .field("vibration_y", data['sensors']['vibration_y']) \
            .field("vibration_z", data['sensors']['vibration_z']) \
            .field("temperature", data['sensors']['temperature']) \
            .field("pressure", data['sensors']['pressure']) \
            .field("rpm", data['sensors']['rpm']) \
            .time(data['timestamp'])
        
        if 'labels' in data:
            point.field("is_degraded", data['labels']['is_degraded'])
            point.field("is_failed", data['labels']['is_failed'])
            point.field("rul_hours", data['labels']['rul_hours'])
        
        write_api.write(bucket=INFLUXDB_BUCKET, record=point)

if __name__ == '__main__':
    write_to_influxdb()
```

---

## Phase 3: Stream Processing (Week 3-4)

### Spark Streaming Job

```python
# spark_streaming_job.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Initialize Spark
spark = SparkSession.builder \
    .appName("PredictiveMaintenance-StreamProcessing") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0") \
    .getOrCreate()

# Define schema
schema = StructType([
    StructField("machine_id", StringType()),
    StructField("timestamp", TimestampType()),
    StructField("sensors", StructType([
        StructField("vibration_x", DoubleType()),
        StructField("vibration_y", DoubleType()),
        StructField("vibration_z", DoubleType()),
        StructField("temperature", DoubleType()),
        StructField("pressure", DoubleType()),
        StructField("rpm", DoubleType())
    ])),
    StructField("metadata", StructType([
        StructField("machine_type", StringType()),
        StructField("location", StringType())
    ]))
])

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9094") \
    .option("subscribe", "sensor-raw") \
    .option("startingOffsets", "earliest") \
    .load()

# Parse JSON
parsed_df = df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# Feature engineering
window_spec_1h = Window.partitionBy("machine_id").orderBy("timestamp").rowsBetween(-360, 0)  # 1 hour
window_spec_6h = Window.partitionBy("machine_id").orderBy("timestamp").rowsBetween(-2160, 0)  # 6 hours

enriched_df = parsed_df \
    .withColumn("vibration_magnitude", sqrt(
        col("sensors.vibration_x")**2 + 
        col("sensors.vibration_y")**2 + 
        col("sensors.vibration_z")**2
    )) \
    .withColumn("vibration_x_rolling_mean_1h", 
        avg("sensors.vibration_x").over(window_spec_1h)) \
    .withColumn("vibration_x_rolling_std_1h", 
        stddev("sensors.vibration_x").over(window_spec_1h)) \
    .withColumn("temperature_rolling_mean_1h", 
        avg("sensors.temperature").over(window_spec_1h)) \
    .withColumn("temperature_rolling_std_1h", 
        stddev("sensors.temperature").over(window_spec_1h)) \
    .withColumn("temperature_rate_of_change",
        col("sensors.temperature") - lag("sensors.temperature", 1).over(
            Window.partitionBy("machine_id").orderBy("timestamp")
        ))

# Write to Kafka enriched topic
query = enriched_df.selectExpr("to_json(struct(*)) AS value") \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9094") \
    .option("topic", "sensor-enriched") \
    .option("checkpointLocation", "/tmp/checkpoint") \
    .start()

query.awaitTermination()
```

---

## Phase 4: ML Model Development (Week 5-10)

### Training Pipeline

```python
# model_training.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.keras

# Load data from InfluxDB
def load_training_data():
    from influxdb_client import InfluxDBClient
    
    client = InfluxDBClient(
        url="http://localhost:8086",
        token="my-super-secret-token",
        org="predictive-maintenance"
    )
    
    query = '''
    from(bucket: "sensors")
      |> range(start: -30d)
      |> filter(fn: (r) => r["_measurement"] == "sensor_readings")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    df = client.query_api().query_data_frame(query)
    return df

# Prepare sequences
def create_sequences(data, sequence_length=100):
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, -1])  # RUL target
    
    return np.array(X), np.array(y)

# Build LSTM model
def build_lstm_model(sequence_length, n_features):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')  # RUL regression
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Training function
def train_model():
    mlflow.set_experiment("predictive-maintenance")
    
    with mlflow.start_run():
        # Load and preprocess data
        df = load_training_data()
        
        # Feature selection
        feature_cols = ['vibration_x', 'vibration_y', 'vibration_z', 
                       'temperature', 'pressure', 'rpm']
        
        # Normalize
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(df[feature_cols + ['rul_hours']])
        
        # Create sequences
        sequence_length = 100
        X, y = create_sequences(data_normalized, sequence_length)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
        
        # Build model
        model = build_lstm_model(sequence_length, len(feature_cols))
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
        
        # Log parameters
        mlflow.log_param("sequence_length", sequence_length)
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("optimizer", "adam")
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=64,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        # Evaluate
        test_loss, test_mae = model.evaluate(X_test, y_test)
        
        # Log metrics
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_mae", test_mae)
        
        # Log model
        mlflow.keras.log_model(model, "model")
        
        print(f"Test MAE: {test_mae:.2f} hours")
        
        return model

if __name__ == '__main__':
    train_model()
```

---

## Phase 5: Model Serving (Week 11-12)

### FastAPI Model Serving

```python
# model_serving_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from typing import List
import redis

app = FastAPI(title="Predictive Maintenance API")

# Load model
model = tf.keras.models.load_model('best_model.h5')

# Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

class SensorData(BaseModel):
    machine_id: str
    sensors: List[List[float]]  # Sequence of sensor readings

class PredictionResponse(BaseModel):
    machine_id: str
    failure_probability: float
    time_to_failure_hours: float
    risk_level: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_failure(data: SensorData):
    try:
        # Prepare input
        X = np.array(data.sensors).reshape(1, -1, 6)
        
        # Make prediction
        prediction = model.predict(X)[0][0]
        
        # Calculate risk metrics
        failure_probability = 1.0 / (1.0 + prediction)  # Convert RUL to probability
        
        risk_level = "LOW"
        if failure_probability > 0.8:
            risk_level = "CRITICAL"
        elif failure_probability > 0.6:
            risk_level = "HIGH"
        elif failure_probability > 0.4:
            risk_level = "MEDIUM"
        
        return PredictionResponse(
            machine_id=data.machine_id,
            failure_probability=failure_probability,
            time_to_failure_hours=float(prediction),
            risk_level=risk_level,
            confidence=0.85
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run the API:
```bash
pip install fastapi uvicorn redis
python model_serving_api.py
```

---

## Phase 6: Monitoring & Alerting (Week 13-14)

### Grafana Dashboard Setup

1. Access Grafana at `http://localhost:3000` (admin/admin)
2. Add InfluxDB data source:
   - URL: `http://influxdb:8086`
   - Token: `my-super-secret-token`
   - Organization: `predictive-maintenance`
   - Bucket: `sensors`

3. Create dashboard panels:
   - Real-time sensor readings (time series)
   - Failure probability gauge
   - Alert count (stat)
   - Machine health status (table)

### Alert System

```python
# alert_system.py
from kafka import KafkaConsumer
import json
import smtplib
from email.mime.text import MIMEText

def send_email_alert(machine_id, risk_level, message):
    # Configure SMTP
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "alerts@company.com"
    
    msg = MIMEText(message)
    msg['Subject'] = f"[{risk_level}] Machine {machine_id} Alert"
    msg['From'] = sender_email
    msg['To'] = "maintenance@company.com"
    
    # Send email (configure credentials)
    # server = smtplib.SMTP(smtp_server, smtp_port)
    # server.starttls()
    # server.sendmail(sender_email, ["maintenance@company.com"], msg.as_string())
    # server.quit()
    
    print(f"Alert sent for {machine_id}: {risk_level}")

def monitor_predictions():
    consumer = KafkaConsumer(
        'predictions',
        bootstrap_servers=['localhost:9094'],
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id='alert-system'
    )
    
    for message in consumer:
        prediction = message.value
        
        if prediction['risk_level'] in ['CRITICAL', 'HIGH']:
            send_email_alert(
                prediction['machine_id'],
                prediction['risk_level'],
                f"Failure probability: {prediction['failure_probability']:.2%}\n"
                f"Time to failure: {prediction['time_to_failure_hours']:.1f} hours"
            )

if __name__ == '__main__':
    monitor_predictions()
```

---

## Testing & Validation

### Integration Test

```python
# test_pipeline.py
import requests
import numpy as np

def test_end_to_end():
    # Generate test data
    test_sequence = np.random.randn(100, 6).tolist()
    
    # Call prediction API
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "machine_id": "TEST-001",
            "sensors": test_sequence
        }
    )
    
    assert response.status_code == 200
    result = response.json()
    
    print("Prediction result:")
    print(f"  Failure probability: {result['failure_probability']:.2%}")
    print(f"  Time to failure: {result['time_to_failure_hours']:.1f} hours")
    print(f"  Risk level: {result['risk_level']}")
    
    assert result['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

if __name__ == '__main__':
    test_end_to_end()
```

---

## Deployment Checklist

- [ ] All infrastructure services running
- [ ] Kafka topics created and verified
- [ ] Database schemas applied
- [ ] Data simulator producing messages
- [ ] Stream processing job running
- [ ] Model trained and registered
- [ ] Model serving API deployed
- [ ] Alert system configured
- [ ] Grafana dashboards created
- [ ] Monitoring configured
- [ ] Integration tests passing
- [ ] Documentation updated

---

## Troubleshooting

### Common Issues

**1. Kafka connection errors**
```bash
# Check Kafka status
docker-compose ps kafka

# View Kafka logs
docker-compose logs kafka
```

**2. InfluxDB write errors**
```bash
# Verify InfluxDB connectivity
curl http://localhost:8086/health

# Check bucket exists
docker exec -it influxdb influx bucket list
```

**3. Model serving errors**
```bash
# Check API logs
docker-compose logs model-api

# Test API health
curl http://localhost:8000/health
```

---

## Performance Optimization

### Tuning Parameters

**Kafka:**
- Increase `num.partitions` for higher throughput
- Adjust `retention.ms` based on storage capacity
- Configure `compression.type=lz4` for better compression

**Spark:**
- Set `spark.executor.memory` based on data volume
- Adjust `spark.sql.shuffle.partitions` for optimal parallelism
- Enable `spark.sql.adaptive.enabled` for dynamic optimization

**Model Inference:**
- Use batch prediction for multiple machines
- Implement model caching
- Consider TensorFlow Serving for production

---

## Next Steps

1. **Enhanced Features**:
   - Implement advanced anomaly detection (Isolation Forest, Autoencoder)
   - Add explainability (SHAP values)
   - Develop ensemble models

2. **Production Hardening**:
   - Add authentication/authorization
   - Implement circuit breakers
   - Set up backup and recovery

3. **Advanced Analytics**:
   - Root cause analysis
   - Maintenance cost optimization
   - Predictive scheduling

4. **Integration**:
   - Connect to existing CMMS
   - Integrate with ERP systems
   - Mobile app development

---

## Resources

- **Documentation**: See `/docs` folder for detailed guides
- **Support**: Contact data-platform@company.com
- **Repository**: https://github.com/company/predictive-maintenance

---

*Last Updated: October 2025*
