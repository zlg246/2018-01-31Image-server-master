import os, glob
import parameters
import shutil
import compileall
import subprocess
import sys
import difflib

ver_stamp = "server_NY." + parameters.ver

# create folders
base_dir = os.path.join(r"C:\Users\Neil\BT\server and distributables", ver_stamp)
master_dir = r"C:\Users\Neil\Documents\GitHub\Image-server"

if os.path.isdir(base_dir):
    print "Removing", base_dir
    shutil.rmtree(base_dir)

dist_dir = os.path.join(base_dir, "dist")
src_dir = os.path.join(base_dir, "src")
os.mkdir(base_dir)
os.mkdir(dist_dir)
os.mkdir(src_dir)
with open(os.path.join(src_dir, "VERSION"), "w") as f:
    f.write(parameters.ver)
shutil.copy(os.path.join(src_dir, "VERSION"),
            os.path.join(dist_dir, "VERSION"))

# find previous version
root = os.path.split(base_dir)[0]
folders = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name)) and name.startswith('server_')]
ver_nums = [[int(v) for v in f.split('.')[-4:]] for f in folders]
pairs = zip(ver_nums, folders)
pairs.sort(key=lambda x:x[0])
folders = [f[1] for f in pairs]
assert os.path.join(root, folders[-1]) == base_dir
prev_version_dir = os.path.join(root, folders[-2])
prev_version = folders[-2].split('_')[1][3:]

# metric and image name diffs
cur_im = os.path.join(master_dir, "%s_images.csv" % parameters.ver)
prev_im = os.path.join(prev_version_dir, 'src', "%s_images.csv" % prev_version)
prev_metric = os.path.join(prev_version_dir, 'src', "%s_metrics.csv" % prev_version)
fn_out = os.path.join(base_dir, 'src', "%s_images_diff.txt" % parameters.ver)
if os.path.isfile(prev_im):
    with open(prev_im) as f_prev:
        with open(cur_im) as f_cur:
            with open(fn_out, 'w') as f_out:
                line_prev = f_prev.readlines()
                line_cur = f_cur.readlines()
                diff = difflib.unified_diff(line_prev, line_cur)
                f_out.write('\n'.join(diff))
else:
    print "Image names for previous version not found"

cur_im = os.path.join(master_dir, "%s_metrics.csv" % parameters.ver)
prev_im = os.path.join(prev_version_dir, 'src', "%s_metrics.csv" % prev_version)
prev_metric = os.path.join(prev_version_dir, 'src', "%s_metrics.csv" % prev_version)
fn_out = os.path.join(base_dir, 'src', "%s_metrics_diff.txt" % parameters.ver)
if os.path.isfile(prev_im):
    with open(prev_im) as f_prev:
        with open(cur_im) as f_cur:
            with open(fn_out, 'w') as f_out:
                line_prev = f_prev.readlines()
                line_cur = f_cur.readlines()
                diff = difflib.unified_diff(line_prev, line_cur)
                diff = [d for d in diff if 'runtime' not in d]
                f_out.write('\n'.join(diff))
else:
    print "Metric names for previous version not found"

# copy everything to source
source_files = ['features_multi_wafer.py', 'features_multi_cell.py',
                'image_processing.py',
                'parameters.py', 'PROTOCOL', 'README',
                'server.py', 'TIFFfile.py', 'pixel_ops.pyx', 'cropping.py',
                'pixel_ops.pyd', #'pthreadGC2.dll', 'tbb.dll',
                'cc_label.pyx', 'features_cz_cell.py',
                'cc_label.pyd', 'features_cz_wafer.py', 'features_block.py',
                'client_NY.py', "cell_processing.py", "colormaps.py",
                'features_x3.py', 'features_module.py',
                'FF.py', 'hash_fft_mask.npy', 'features_slugs.py',
                'features_perc.py', 'features_resolution.py',
                'tool_calibrate.py', 'features_stripes.py', 'features_qc_c3.py',
                "%s_images.csv" % parameters.ver,
                "%s_metrics.csv" % parameters.ver]
dist_files = ['cc_label.pyd', 'features_block.pyc', 'cropping.pyc', 'features_multi_cell.pyc',
              'features_cz_wafer.pyc', 'features_multi_wafer.pyc', 'features_cz_cell.pyc',
              'image_processing.pyc', 'parameters.py', 'pixel_ops.pyd', 'features_x3.pyc',
              'server.pyc', 'TIFFfile.pyc', 'features_resolution.pyc', 'features_qc_c3.pyc',
              'hash_fft_mask.npy', 'FF.pyc', 'features_slugs.pyc', 'features_stripes.pyc',
              'features_multi_cell.pyc', "cell_processing.pyc", 'features_perc.pyc',  "colormaps.pyc",
              'features_module.pyc', 'tool_calibrate.pyc']
for s_file in source_files:
    shutil.copy(os.path.join(master_dir, s_file),
                os.path.join(src_dir, s_file))

# get latest release report
reports = glob.glob(r"C:\Users\Neil\BT\release_reports\NY_%s*" % (parameters.ver))
if len(reports) > 0:
    reports.sort()
    shutil.copy(reports[-1], os.path.join(src_dir, os.path.split(reports[-1])[1]))
else:
    print "WARNING: No report found"

if False:
    # generate documentation
    # note: C:\Users\Neil\BT\docs doesn't seem to have made it onto this laptop. check dell.
    auto_docs = r"C:\Users\Neil\Dropbox (Personal)\BT\src\build_docs\_build\latex\ImageProcessing.pdf"
    if os.path.isfile(auto_docs):
        os.remove(auto_docs)
    subprocess.call("generate_docs.bat")
    if not os.path.isfile(auto_docs):
        print "ERROR creating documentation"
        sys.exit()
    shutil.copy(auto_docs, os.path.join(src_dir, "Code documentation.pdf"))

# compile everything
compileall.compile_dir(src_dir, force=1)

for fn in dist_files:
    fn_full = os.path.join(src_dir, fn)
    shutil.copy(fn_full, os.path.join(dist_dir, fn))


