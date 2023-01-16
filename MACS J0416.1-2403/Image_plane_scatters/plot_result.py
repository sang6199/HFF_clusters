import numpy as np
import matplotlib.pyplot as plt
from load_preprocessing import max_m_source, galaxy_m_y, galaxy_m_x, strong_source_number
from load_paramter import interpol_size, kapmap_size

# compute location of lens plane images (pixel scale)
i = 0
for I in range(max_m_source):
    minimized_result = np.load('./lens_rms_result/multiple_image_{}.npy'.format(I))
    source_indexing = np.where(strong_source_number == I)
    gal_m_x = galaxy_m_x[source_indexing]
    gal_m_y = galaxy_m_y[source_indexing]

    f = open('./lens_rms_result/multiple_image_plane_{}.txt'.format(I), 'w')
    for J in range(len(gal_m_x)):
        x_min_indexing = np.argmin(np.abs(gal_m_x[J] - minimized_result[:,0]))
        y_min_indexing = np.argmin(np.abs(gal_m_y[J] - minimized_result[:,1]))
        xmin = minimized_result[x_min_indexing,0]
        ymin = minimized_result[y_min_indexing,1]
        f.write('{} {} {}\n'.format(J, xmin, ymin))
    f.close()


# plot result
pix2arc = (interpol_size / kapmap_size) * 0.05
result_array = np.zeros((len(galaxy_m_x)))
result_dif, result_dif_x, result_dif_y = [], [], []

for I in range(max_m_source):
    source_indexing = np.where(strong_source_number == I)
    data = np.loadtxt('./lens_rms_result/multiple_image_plane_{}.txt'.format(I)).T

    for K in range(len(source_indexing[0])):
        relens_x_coord, relens_y_coord = data[1,K], data[2,K]
        real_x_coord, real_y_coord = galaxy_m_x[source_indexing][K], galaxy_m_y[source_indexing][K]

        dif = (np.abs(real_x_coord*pix2arc - relens_x_coord*pix2arc))**2 + (np.abs(real_y_coord*pix2arc - relens_y_coord*pix2arc))**2
        result_dif.append(dif.tolist())
        result_dif_x.append((real_x_coord * pix2arc - relens_x_coord * pix2arc).tolist())
        result_dif_y.append((real_y_coord * pix2arc - relens_y_coord * pix2arc).tolist())

np.save('rms_macsj0416.npy', np.sqrt(result_dif))
total_rms = np.sqrt(np.sum(np.array(result_dif))/(len(galaxy_m_x)))
print('result_rms:',total_rms)

plt.figure(1, figsize=(6,6))
plt.hist(np.array(np.sqrt(result_dif)), bins=np.arange(0, np.max(np.round(np.array(np.sqrt(result_dif)), 4)) + 0.001, 0.001))
plt.title('Total $\Delta_{rms}$:'+' {:.5f} arcsec'.format(total_rms), fontsize=12)
plt.xlabel('$\Delta$ in arcsec', fontsize=12)
plt.ylabel('Number of images', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('rms_macsj0416.pdf')


plt.figure(2, figsize=(6,6))

left, width = 0.15, 0.6
bottom, height = 0.15, 0.6
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

ax0 = plt.axes(rect_scatter)
ax1 = plt.axes(rect_histx)
ax2 = plt.axes(rect_histy)

ax0.axvline(x=0, ymax=1, ymin=-1, c='black', linewidth=0.8)
ax0.axhline(y=0, xmax=1, xmin=-1, c='black', linewidth=0.8)
ax0.scatter(result_dif_x, result_dif_y, edgecolors='black')
ax0.set_xlabel('$\Delta_{x}$ in arcsec', fontsize=12)
ax0.set_ylabel('$\Delta_{y}$ in arcsec', fontsize=12)
ax0.set_xlim([-(np.max(np.abs(np.array([result_dif_x, result_dif_y]))) * 1.2), (np.max(np.abs(np.array([result_dif_x, result_dif_y]))) * 1.2)])
ax0.set_ylim([-(np.max(np.abs(np.array([result_dif_x, result_dif_y]))) * 1.2), (np.max(np.abs(np.array([result_dif_x, result_dif_y]))) * 1.2)])

ax1.axvline(x=0, ymin=0, ymax=30, c='black', linewidth=0.8)
ax1.hist(np.array(result_dif_x), bins=np.arange(np.min(np.round(np.array(result_dif_x), 4)) - 0.001, np.max(np.round(np.array(result_dif_x), 4)) + 0.001, 0.002))
ax1.set_xticks([])
ax1.set_xlim([-(np.max(np.abs(np.array([result_dif_x, result_dif_y]))) * 1.2), (np.max(np.abs(np.array([result_dif_x, result_dif_y]))) * 1.2)])
ax1.set_ylabel('N')

ax2.axhline(y=0, xmin=0, xmax=30, c='black', linewidth=0.8)
ax2.hist(np.array(result_dif_y), bins=np.arange(np.min(np.round(np.array(result_dif_y), 4)) - 0.001, np.max(np.round(np.array(result_dif_y), 4)) + 0.001, 0.002), orientation='horizontal')
ax2.set_ylim([-(np.max(np.abs(np.array([result_dif_x, result_dif_y]))) * 1.2), (np.max(np.abs(np.array([result_dif_x, result_dif_y]))) * 1.2)])
ax2.set_yticks([])
ax2.set_xlabel('N')
ax0.text(0.95, 0.05, '$\Delta_{rms}$:'+' {:.4f}\'\''.format(total_rms), verticalalignment='bottom', horizontalalignment='right',
        transform=ax0.transAxes,fontsize=13)
plt.savefig('rms_macsj0416_2d.pdf')

result_dif_x_a370 = np.array([result_dif_x])
result_dif_y_a370 = np.array([result_dif_y])
result_dif = np.concatenate([result_dif_x_a370, result_dif_y_a370], axis=0)
np.save('rms_macsj0416_2d.npy', result_dif)
