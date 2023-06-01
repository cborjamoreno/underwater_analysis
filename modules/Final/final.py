import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import time

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from modules.Module3D.depth_estimation import estimate
from modules.Segmentation.segmentation import segmentationSAM, binarySegmentationDepth, floatingSegmentation

LIGHT_PURPLE = (213, 184, 255)
DARK_BLUE = (1, 1, 122)

def applyMask(mask, points, coloring, img=None):
    """Apply skyline mask to pointcloud

    Parameters
    ----------
    mask : array_like, shape (nrows, ncols, 3)
        Image that represent de skyline mask
    points : array_like, shape (N,3)
        Array containing the set of points in space

    Returns
    -------
    point_array : array_like 
        Resulting array after deleting water pixels

    """
    nrows,ncols = points.shape
    
    points_mask = points.copy()
    
    delete_counter = 0
    
    for i in range(nrows):
        for j in range(ncols):
            if mask[i,j,:].tolist() == list(LIGHT_PURPLE):
                points_mask[i,j] = 1
                delete_counter += 1
                
    useful = nrows*ncols - delete_counter
                
    point_array = np.zeros(shape=(useful,3))
    colors = np.array(np.zeros(shape=(useful,3)))
    
    i = 0
    for x in range(nrows):
        for y in range(ncols):
            if points_mask[x,y] < 0.9:
                point_array[i] = [x,y,points_mask[x,y]]
                if coloring == 'FLOATING':
                    if img is not None:
                        if mask[x,y,:].tolist() == list((0,128,90)):
                            colors[i] = [val/255.0 for val in list((255,0,0))]
                        else:
                            colors[i] = [val/255.0 for val in list(img[x,y,:])]
                i += 1
    
    if coloring == 'DEPTH':
        colors = point_array[:, 2]
    
    return point_array, colors

def showPointcloudWithMask(depth, mask, coloring, img=None):
    """Shows pointcloud from depth points after apply the segmentation mask 'mask' to delete water points with the specify coloring type. 

    Parameters
    ----------
    depth : array_like, shape (N,3)
        Array containing the set of points in space
    mask : array_like, shape (nrows, ncols, 3)
        Floating segmentation mask
    coloring : str, {'OBJECTS', 'FLOATING', 'DEPTH'}
        Points coloring type.
         - FLOATING: each point p is colored with RGB = (255, 0, 0) if p is part of floating objetc in mask or with the original color in img.
         - DEPTH: each point (x,y,z) is colored with colormap 'jet_r' taking depth[x,y,z] value.
    """

    if mask is None:
        print('ERROR: mask is empty. Try to use another mask')
        return
    

    # Resize mask
    nrows,ncols = depth.shape
    mask_resized = cv2.resize(mask, (ncols,nrows), interpolation = cv2.INTER_AREA)
    img_resized = cv2.resize(img, (ncols,nrows), interpolation = cv2.INTER_AREA)

    # Apply mask to pointcloud to delete water points
    pc_mask, colors = applyMask(mask_resized, depth, coloring, img_resized)

    # Plot pointcloud
    fig = plt.figure().add_subplot(projection='3d')
    fig.set_title('3D pointcloud')
    
    if coloring == 'FLOATING':
        fig.scatter(
            pc_mask[:, 1],
            pc_mask[:, 2],
            pc_mask[:, 0],
            s=0.03,
            c=colors,
            vmin=np.amin(depth[:,2]),
            vmax = np.amax(depth[:,2])
        )
    else:
        cmap="jet_r"
        fig.scatter(
            pc_mask[:, 1],
            pc_mask[:, 2],
            pc_mask[:, 0],
            s=0.03,
            c=colors,
            cmap=cmap,
            vmin=np.amin(depth[:,2]),
            vmax = np.amax(depth[:,2])
        )
        cmap = mpl.cm.jet_r
        norm = mpl.colors.Normalize(vmin=np.amin(depth[:,2]), vmax=np.amax(depth[:,2]))

        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),label='depth estimation value')
    fig.view_init(15,235)

    fig.set_xlim3d(0, ncols)
    fig.set_ylim3d(np.amin(depth[:,2]), np.amax(depth[:,2]))
    fig.set_zlim3d(0, nrows)

    fig.set_xlabel(" Y ")
    fig.set_ylabel(" Z ")
    fig.set_zlabel(" X ")

    fig.invert_zaxis()
    plt.show()

def showFinalSegmentation(binary_mask, color_mask=None):
    """Shows both binary and object segmentation images.

    Parameters
    ----------
    binary_mask : array_like, shape (nrows, ncols, 3)
        Binary segmentation mask
    color_mask : array_like, shape (nrows, ncols, 3)
        Object segmentation mask. If color_mask is None, object segmentation is not shown

    """

    # Resize mask
    nrows, ncols,_ = binary_mask.shape
    binaryMask_resized = cv2.resize(binary_mask, (ncols,nrows), interpolation = cv2.INTER_AREA)



    if color_mask is None:
        ncols_plot = 1

    else:
        colorMask_resized = cv2.resize(color_mask, (ncols,nrows), interpolation = cv2.INTER_AREA)
        ncols_plot = 2

    legend_data = [
        [127,list(LIGHT_PURPLE),"agua"],
        [126,list(DARK_BLUE),"escena"]
    ]
    handles = [
        Rectangle((0,0),1,1, color = [v/255 for v in c]) for k,c,n in legend_data
    ]
    labels = [n for k,c,n in legend_data]


    # Plot segmentation
    fig = plt.figure(figsize=plt.figaspect(0.5))
        
    # ===========================
    # First subplot (binary mask)
    # ===========================
    # set up the axes for the first plot
    ax = fig.add_subplot(1, ncols_plot, 1)
    ax.set_title('Binary segmentation')
    ax.imshow(binaryMask_resized)

    ax.grid(False)
    ax.axis('off')
    ax.legend(handles,labels)

    if color_mask is not None:

        # ===========================
        # Second subplot (color mask)
        # ===========================
        # set up the axes for the second plot
        ax = fig.add_subplot(1, ncols_plot, 2)
        ax.set_title('Object segmentation')
        ax.imshow(colorMask_resized)

        ax.grid(False)
        ax.axis('off')
        ax.legend([handles[0]],[labels[0]])
    
    plt.show()


def segmentationFinal(image_path,coloring):
    """Get the segmentation combining SAM with depth estimation. If possible, the function returns an object segmentation

    Parameters
    ----------
    image_path : str
        Image path

    Returns
    -------
    binary_mask : array_like
        Image with binary segmentation. If binary mask calculated with SAM does not segment water properly, depth based segmentation will be used.
    color_mask : array_like
        Image with object segmentation. If binary mask calculated with SAM does not segment water properly, color_mask will be None
    """


    def getIntersectAndUnion(merged, segment_index, thresh):
        """Calculates intersection and union values between 'segment' and 'thresh'"""

        intersection = 0
        union = 0

        
        # Calculate intersection and union values
        for i in range(merged.shape[0]):
            for j in range(merged.shape[1]):
                if merged[i,j] == segment_index+1:
                    if thresh[i,j,:].tolist() == list(LIGHT_PURPLE):
                        intersection += 1
                    union += 1

        return intersection, union
    
    start = time.time()
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    nrows_img, ncols_img, _ = img.shape

    # Estimate depth for image
    depth = estimate(image_path)

    # Segmentation with depth
    thresh = binarySegmentationDepth(depth)
    thresh = cv2.resize(thresh, (ncols_img,nrows_img), interpolation = cv2.INTER_LINEAR_EXACT)
    
    
    # Count water pixels 
    water = 0
    for i in range(nrows_img):
        for j in range(ncols_img):
            if thresh[i,j,:].tolist() == list(LIGHT_PURPLE):
                water += 1
    water_percent = water/(nrows_img*ncols_img)
    
    # Get SAM segmentation
    masks = segmentationSAM(img)
    masks = sorted(masks, key=(lambda x: x['area']))

    new_areas = np.zeros(len(masks)).tolist()

    # Merge masks
    merged = np.zeros((nrows_img, ncols_img),dtype=int)
    for i in range(merged.shape[0]):
        for j in range(merged.shape[1]):
            for m in range(len(masks)):
                if masks[m]['segmentation'][i,j]:
                    merged[i,j] = m+1
                    new_areas[m] += 1
                    break
    
    # Select biggest segment as water
    water_segment_index = np.argmax(np.array(new_areas))
    
    intersection, union = getIntersectAndUnion(merged, water_segment_index, thresh)
    
    # Check if selected water segment has a good "intersect over union" value
    while (intersection/union) < 0.5:
        # "Intersect over union" value is not good enough. Search for another water segment
        del new_areas[water_segment_index]
        if len(new_areas) == 0:
            print('Intersection over union is not good enough for any mask. Segmentation based on depth estimation will be used')
            end = time.time()
            print('Execution time:',end-start)
            showFinalSegmentation(thresh)

            if coloring == 'FLOATING':
                thresh = thresh.astype('uint8')
                floating_mask = floatingSegmentation(thresh)
                showPointcloudWithMask(depth,floating_mask,coloring,img)
            else:
                thresh = thresh.astype('uint8')
                showPointcloudWithMask(depth,thresh,coloring,img)
            return thresh, None
        water_segment_index = np.argmax(np.array(new_areas))
        
        intersection, union = getIntersectAndUnion(merged, water_segment_index, thresh)
        
    union += (water - intersection)

    color_mask = img.copy()
    binary_mask = img.copy()
    
        
    # Defining random colors
    colors = []
    for i in range(len(masks)):
        for j in range(3):
            colors.append(list(np.random.choice(range(255),size=3)))
    
    water = 0

    # Create binary and color masks
    for i in range(merged.shape[0]):
        for j in range(merged.shape[1]):
            if merged[i,j] == water_segment_index+1:
                color_mask[i,j] = LIGHT_PURPLE
                binary_mask[i,j] = LIGHT_PURPLE
                water +=1
            else:  
                color_mask[i,j] = colors[merged[i,j]]
                binary_mask[i,j] = DARK_BLUE

    water_percent_SAM = water/(merged.shape[0]*merged.shape[1])

    print(type(binary_mask[0,0,0]))
    print(type(thresh[0,0,0]))

    success = True
    if water_percent_SAM/water_percent < 0.5:
        print('No se ha encontrado una máscara binaria mejor. Se utiliza la calculada a partir de la estimación de profundidad')
        end = time.time()
        print('Execution time:',end-start)
        success = False
    
    end = time.time()
    print('Execution time:',end-start)

    if success == False:
        thresh = thresh.astype('uint8')
        binary_mask = thresh.copy()
        color_mask = None
    showFinalSegmentation(binary_mask,color_mask)

    if coloring == 'FLOATING':
        floating_mask = floatingSegmentation(binary_mask)
        showPointcloudWithMask(depth,floating_mask,coloring,img)
    else:
        showPointcloudWithMask(depth,binary_mask,coloring,img)
    
    return binary_mask, color_mask

