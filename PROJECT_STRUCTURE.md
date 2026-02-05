├── 📄 Root Configuration Files
│   ├── .env                           
│   ├── config.yaml                    
│   ├── config.example.yaml            
│   ├── env.example                    
│   ├── docker-compose.yml             
│   ├── docker-compose.airflow.yml     
│   ├── docker-compose.kafka.yml       
│   ├── Makefile                       
│   ├── pytest.ini                     
│   ├── requirements.txt               
│   ├── requirements-mlflow.txt        
│   ├── setup.py                       
│   ├── README.md                      
│   ├── .gitignore                     
│   ├── run_local.sh                   
│   ├── run_ecs.sh                     
│   ├── cleanup_docker.sh              
│   └── make_rds.sh                    
│
├── 🐳 docker/                         
│   ├── Dockerfile.airflow             
│   ├── Dockerfile.base                
│   ├── Dockerfile.kafka-analytics     
│   ├── Dockerfile.kafka-inference     
│   ├── Dockerfile.kafka-producer      
│   ├── Dockerfile.mlflow              
│   ├── entrypoint-template.sh         
│   ├── mlflow-entrypoint.sh           
│   ├── streamlit-entrypoints.sh       
│   └── airflow/dags/                  
│       ├── data_pipeline_ecs_dag.py
│       └── train_pipeline_ecs_dag.py
│
├── ☁️ ecs-deployment/                 
│   ├── 00_env.sh.example              
│   ├── 00_env.sh                      
│   ├── 10_bootstrap.sh                
│   ├── 20_networking.sh               
│   ├── 30_iam.sh                      
│   ├── 40_cluster_alb.sh              
│   ├── 50_register_tasks.sh           
│   ├── 60_services.sh                 
│   ├── 70_airflow_init.sh             
│   ├── 80_airflow_vars.sh             
│   ├── 90_cleanup_all.sh              
│   ├── rebuild_for_amd64.sh           
│   ├── restart_ecs.sh                 
│   ├── stop_ecs.sh                    
│   ├── .env.out                       
│   ├── airflow/dags/                  
│   │   ├── data_pipeline_ecs_dag.py
│   │   └── train_pipeline_ecs_dag.py
│   └── taskers/                       
│       ├── airflow-scheduler.json.template
│       ├── airflow-web.json.template
│       ├── airflow-worker.json.template
│       ├── data-pipeline.json.template
│       ├── inference-pipeline.json.template
│       ├── kafka-analytics.json.template
│       ├── kafka-broker.json.template
│       ├── kafka-inference.json.template
│       ├── kafka-producer.json.template
│       ├── mlflow-tracking.json.template
│       └── train-pipeline.json.template
│
├── 🌊 airflow/                        
│   ├── dags/                          
│   │   ├── data_pipeline_dag.py       
│   │   └── model_training_dag.py      
│   ├── logs/                          
│   └── plugins/                       
│
├── 📊 kafka/                          
│   ├── producer_service.py            
│   ├── inference_service.py           
│   └── analytics_service.py           
│
├── 🔧 pipelines/                      
│   ├── __init__.py
│   ├── data_pipeline.py               
│   ├── training_pipeline.py           
│   └── inference_pipeline.py          
│
├── 💻 src/                            
│   ├── __init__.py
│   ├── data_ingestion.py              
│   ├── handle_missing_values.py       
│   ├── outlier_detection.py           
│   ├── feature_scaling.py             
│   ├── feature_encoding.py            
│   ├── feature_binning.py             
│   ├── data_splitter.py               
│   ├── model_building.py              
│   ├── model_training.py              
│   ├── model_evaluation.py            
│   └── model_inference.py             
│
├── 🛠️ utils/                          
│   ├── __init__.py
│   ├── config.py                      
│   ├── s3_io.py                       
│   ├── s3_artifact_manager.py         
│   ├── artifact_manager.py            
│   ├── mlflow_utils.py                
│   ├── kafka_utils.py                 
│   ├── db_manager.py                  
│   ├── timestamp_resolver.py          
│   ├── spark_session.py               
│   └── spark_utils.py                 
|
|
├── 🧪 tests/                          
│   ├── unit/                          
│   │   ├── __init__.py
│   │   ├── test_data_ingestion.py
│   │   └── test_feature_engineering.py
│   ├── integration/                   
│   │   ├── __init__.py
│   │   ├── test_kafka_flow.py
│   │   └── test_pipeline_flow.py
│   ├── validate_data.py               
│   ├── validate_model_simple.py       
│   └── README.md                      
│
│
├── 📦 artifacts/                      
│   ├── data/                          
│   │   ├── X_train.pkl
│   │   ├── X_test.pkl
│   │   ├── y_train.pkl
│   │   ├── y_test.pkl
│   │   ├── test_data.pkl
│   │   ├── scaler.pkl
│   │   └── encoders.pkl
│   ├── models/                        
│   │   └── best_model.pkl
│   └── analytics_reports/             # Generated reports
│       ├── churn_trends_*.csv
│       ├── demographics_*.csv
│       ├── geography_analysis_*.csv
│       ├── high_risk_customers_*.csv
│       ├── model_performance_*.csv
│       ├── overall_summary_*.csv
│       ├── realtime_dashboard_*.csv
│       └── recent_predictions_*.csv
│
├── 📊 data/                           
│   ├── raw/
│       ├── ChurnModelling.csv             
│       └── ChurnModelling_Clean.csv       
│
├── 📋 reports/                        
│   ├── data_validation_report.json    
│   └── model_validation_report.json  
│
├── 🔨 scripts/                        
│   ├── prepare_clean_dataset.py       
│   └── analytics_visualization.ipynb  
│
├── 🗄️ sql/                            
│   └── create_analytics_tables.sql    
│
├── 🖼️ image/                          
│   └── Makefile/                      
│       ├── 1761108669361.png
│       ├── 1761108674558.png
│       └── 1761108681365.png
│
├── 🔄 .github/                        
│   └── workflows/
│       ├── ci.yml                     
│       └── dependabot.yml             
│
├── 🐍 .venv/                          
│   ├── bin/                          
│   ├── lib/                           
│   ├── include/                      
│   └── share/                         
```
