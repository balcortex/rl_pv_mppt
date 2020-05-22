import matlab.engine

print("Starting engine")
eng = matlab.engine.start_matlab()

# eng.eval("model='pvmppt/matlab_model'", nargout=0)
# eng.eval("load_system(model)", nargout=0)
# status = eng.eval("get_param('matlab_model', 'SimulationStatus')", nargout=1)
# print(status)

eng.eval("x=2", nargout=0)
eng.eval("y=x*2", nargout=0)
eng.eval("z=y*2", nargout=0)
print(eng.workspace)
