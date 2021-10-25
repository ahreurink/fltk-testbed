DOCKER_BUILDKIT=1 docker build . --tag gcr.io/qpe21-325513/fltk
docker push gcr.io/qpe21-325513/fltk
cd charts
helm uninstall orchestrator -n test
helm install orchestrator ./orchestrator --namespace test -f fltk-values.yaml
