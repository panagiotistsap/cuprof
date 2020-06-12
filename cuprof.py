try:
    from pycuda.compiler import SourceModule
    from pycuda import gpuarray, driver as drv, tools
    import atexit      
    from blond.utils.bmath import gpu_num
    prev_device_number = gpu_num
    drv.init()
    dev = drv.Device(gpu_num)
except:
    pass
from joblib import dump,load
from time import time
import matplotlib.pyplot as plt


time_dict = {}



calls_dict = {}
prof_enable = None

stack_with_functions = []




class region_timer(object):
    def __init__(self, region_name, region_kind='cpu'): 
        global prof_enable
        if (not prof_enable):
            return
        self.region_name = region_name
        self.region_kind = region_kind
        self.elapsed_time = 0
        
        if (self.region_kind not in ['cpu','gpu']):
            print(self.region_kind)
            print("region kind not supported\nExiting......")
            exit(1)

    def __enter__(self): 
        global stack_with_functions,prof_enable
        if (not prof_enable):
            return
        if (len(stack_with_functions)>0):
            stack_with_functions[-1].stop()
        if (self.region_kind == 'gpu'):
            self.start = drv.Event()
            self.end   = drv.Event()
            self.start.record()
        else:
            self.start = time()
        stack_with_functions.append(self)

    def __exit__(self, exc_type, exc_value, tb): 
        global prof_enable
        global stack_with_functions
        if (not prof_enable):
            return
        if (self.region_kind == 'gpu'):
            self.end.record()
            self.end.synchronize()
            self.elapsed_time += self.start.time_till(self.end)*1e-3
        else:
            self.elapsed_time += time() - self.start
        #### update dict
        del stack_with_functions[-1]
        if (len(stack_with_functions)>0):
            stack_with_functions[-1].start_again()
        if (not prof_enable):
            return
        if (self.region_name not in time_dict):
            time_dict[self.region_name] = self.elapsed_time
            calls_dict[self.region_name] = 1
        else:
            time_dict[self.region_name] += self.elapsed_time
            calls_dict[self.region_name] += 1

    def stop(self):
        if (not prof_enable):
            return
        if (self.region_kind == 'gpu'):
            self.end.record()
            self.end.synchronize()
            self.elapsed_time += self.start.time_till(self.end)*1e-3
        else:
            self.elapsed_time += time() - self.start
        
    def start_again(self):
        if (not prof_enable):
            return
        if (self.region_kind == 'gpu'):
            self.start = drv.Event()
            self.end   = drv.Event()
            self.start.record()
        else:
            self.start = time()


def time_decorator(function_name,function_kind='cpu'):
    global prof_enable
    def real_decorator(function):
        def wrapper(*args, **kwargs):
            if (prof_enable):
                before = region_timer(region_name=function_name, region_kind=function_kind)
                before.__enter__()
            res = function(*args, **kwargs)
            if (prof_enable):
                before.__exit__(None,None,None)
            return res
        return wrapper

    return real_decorator


def report():
    print('{:<14}  {:<20}  {:<20}'.format("function", "total_time","number of calls"))
    for i in time_dict:
        print('{:<14}  {:<20}  {:<20}'.format(i, time_dict[i],calls_dict[i]))
    print('{:<14}  {:<20}'.format("total time: ", sum(time_dict.values())))
    #print("total time: ", sum(time_dict.values()))


def enable():
    global prof_enable
    prof_enable = True


def disable():
    global prof_enable
    prof_enable = False


def save_report(filename):
    global time_dict
    dump(time_dict, filename)


def create_pie_chart(name):
    global time_dict
    labels = [i for i in time_dict]
    sizes = [time_dict[i] for i in labels]
    #print(sizes)
    
    patches, texts  = plt.pie(sizes, shadow = True)
    plt.legend(patches, labels, loc="lower center", ncol=3)
    plt.title("Statistics")
    plt.axis('equal')
    plt.savefig(name)

curr_dict = {}
init_dict = curr_dict
curr_dict['name'] = 'main_point'

class region_timer_new(object):
    def __init__(self, region_name, region_kind='cpu'): 
        global prof_enable,curr_dict
        if (not prof_enable):
            return
        self.region_name = region_name
        if (region_name not in curr_dict):
            curr_dict[region_name] = {}
            new_dict = curr_dict[region_name]
            new_dict['name'] = self.region_name
            new_dict['time'] = 0.
            new_dict['calls'] = 0
            new_dict['prev'] = curr_dict
            new_dict['self'] = self
        else:
            new_dict = curr_dict[region_name]
            new_dict['self'] = self

    def __enter__(self): 
        global curr_dict,prof_enable
        if (not prof_enable):
            return
        self.stop()
        curr_dict = curr_dict[self.region_name]

        self.start = time()


    def __exit__(self, exc_type, exc_value, tb): 
        global prof_enable,curr_dict
        if (not prof_enable):
            return
        curr_dict['time'] += time() - self.start
        curr_dict['calls'] += 1
        curr_dict = curr_dict['prev']
        try:
            curr_dict['self'].start_again()
        except:
            pass

    def stop(self):
        if (not prof_enable):
            return
        if (curr_dict['name']!='main_point'):
            curr_dict['time'] += time() - curr_dict['self'].start

    def start_again(self):
        if (not prof_enable):
            return
        self.start = time()


def time_decorator_new(function_name, function_kind = None):
    global prof_enable
    def real_decorator_new(function):
        def wrapper(*args, **kwargs):
            if (prof_enable):
                before = region_timer_new(region_name=function_name)
                before.__enter__()
            res = function(*args, **kwargs)
            if (prof_enable):
                before.__exit__(None,None,None)
            return res
        return wrapper

    return real_decorator_new


_time = 0
def report_new():
    global curr_dict,_time
    _time = 0
    helping_report_new(curr_dict)
    print(_time)

depth = 0
def helping_report_new(my_dict):
    global depth,_time
    values = [i for i in my_dict]
    print(depth*"  ",my_dict['name'])
    try:
        _time  += my_dict['time']
        print(depth*"  ","time -->", my_dict['time'])
    except:
        pass
    for v in ['name', 'time', 'calls', 'self', 'prev']:
        if (v in values):
            values.remove(v)
    #print(depth*"  ",values)
    depth += 2
    for v in values:
        helping_report_new(my_dict[v])
    
    
    depth -= 2

    
def use_new():
    time_decorator = time_decorator_new
    region_timer = region_timer_new


