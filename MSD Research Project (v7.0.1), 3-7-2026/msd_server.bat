@set py_port=80
@set msd_host=localhost
@set msd_port=8082

@rem Run MSD Server
@start java -Xmx6G -cp bin msd_server.MSDServer %msd_host% %msd_port%

@rem Run HTTP Server
@rem @start python -m http.server %py_port%

@rem Open brownser
@rem @start http://localhost:%py_port%/src/msd_builder/
