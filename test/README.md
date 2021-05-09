## Testing
```bash
conda install pytest
mkdir ../s3
pytest
```

For developers and debugging:
```bash
mkdir ../s3
python3 test/test_netcdf2d_func.py
python3 test/test_netcdf2d1.py
python3 test/test_netcdf2d2.py
python3 test/test_netcdf2d3.py
python3 test/test_netcdf2d4.py
python3 test/test_netcdf2d1.py && python3 test/test_netcdf2d2.py && python3 test/test_netcdf2d3.py && python3 test/test_netcdf2d4.py
```