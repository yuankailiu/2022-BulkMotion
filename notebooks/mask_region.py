import h5py
from mintpy.utils import readfile

mask = '/marmot-nobak/ykliu/aqaba/topsStack/d021/mintpy/maskTempCoh_high.h5'
vtmp = '/marmot-nobak/ykliu/aqaba/topsStack/d021/mintpy/velocity_out/velocity2.h5'


mm = readfile.read(mask)[0]
atr = readfile.read(mask)[1]
vv = readfile.read(vtmp, datasetName='velocity')[0] * 1e3

vv[1*mm==0] = np.nan
vv -= np.nanmedian(vv)

tmp = np.ones_like(vv)
tmp[:1250,:] = 0
mn = np.array(tmp * [vv<-2.][0], dtype=bool)
mn = np.invert(mn) * mm


plt.figure(figsize=[8,8])
im = plt.imshow(mm, interpolation='nearest')
plt.colorbar(im)
plt.show()


plt.figure(figsize=[8,8])
im = plt.imshow(mn, interpolation='nearest')
plt.colorbar(im)
plt.show()


vv[1*mn==0] = np.nan
vv -= np.nanmedian(vv)

plt.figure(figsize=[8,8])
im = plt.imshow(vv, interpolation='nearest', vmin=-5, vmax=5, cmap='RdBu_r')
plt.colorbar(im)
plt.show()



# save velocity
outfile = '/marmot-nobak/ykliu/aqaba/topsStack/d021/mintpy/maskTempCoh_highMsk.h5'

with h5py.File(outfile,'w') as out_data:
    ds = out_data.create_dataset('mask', shape=mn.shape, dtype=mn.dtype)
    ds[:] = mn
    for key in atr.keys():
        out_data.attrs[key] = str(atr[key])