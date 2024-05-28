import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import time
import colorsys

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from modules.Module3D.depth_estimation import estimate
from modules.Segmentation.segmentation import segmentationSAM, binarySegmentationDepth, floatingSegmentation, showBinarySegmentationDepth

LIGHT_PURPLE = (213, 184, 255)
DARK_BLUE = (1, 1, 122)
GREEN = (0, 128, 90)

def applyMask(mask, depth, coloring, img=None):
    """Apply segmentation mask to pointcloud

    Parameters
    ----------
    mask : array_like, shape (nrows, ncols, 3)
        Segmentation mask. If coloring = 'FLOATING' mask should be a floating segmentation mask.
    depth : array_like, shape (nrows,ncols)
        Depth estimation
    coloring : str, {'FLOATING', 'DEPTH'}
        Points coloring type.
         - FLOATING: each point p is colored with RGB = (255, 0, 0) if p is part of floating objetc in mask or with the original color in img.
         - DEPTH: each point (x,y,z) is colored with colormap 'jet_r' taking depth[x,y,z] value.
    img : array_like, shape (rows,cols,3), optional
        Original image. If coloring = 'FLOATING', img can't be None.
         

    Returns
    -------
    point_array : array_like 
        Resulting array after deleting water pixels

    """
    nrows,ncols = depth.shape
    points_mask = depth.copy()
    
    delete_counter = 0
    
    for i in range(nrows):
        for j in range(ncols):
            if mask[i,j,:].tolist() == list(LIGHT_PURPLE) or depth[i,j] > 0.99:
                points_mask[i,j] = 255
                delete_counter += 1
    
    useful = nrows*ncols - delete_counter
    point_array = np.zeros(shape=(useful,3))
    colors = np.array(np.zeros(shape=(useful,3)))

    print('Deleted',delete_counter,'water pixels')
    print('Useful',useful,'pixels')
    
    i = 0
    red = 0
    for x in range(nrows):
        for y in range(ncols):
            if points_mask[x,y] != 255:
                point_array[i] = [x,y,points_mask[x,y]]
                if coloring == 'FLOATING':
                    assert img is not None, "img can't be None if coloring = 'FLOATING"
                    if mask[x,y,:].tolist() == list((0,128,90)):
                        colors[i] = [val/255.0 for val in list((255,0,0))]
                        red +=1
                    else:
                        colors[i] = [val/255.0 for val in list(img[x,y,:])]
                i += 1
    print('i:',i)
    print('red:',red)
    if coloring == 'DEPTH':
        colors = point_array[:, 2]
    
    return point_array, colors

def showPointcloudWithMask(depth, mask, coloring, img=None, output_path=None):
    """Shows pointcloud from depth points after apply the segmentation mask 'mask' to delete water points with the specify coloring type. 

    Parameters
    ----------
    depth : array_like, shape (N,3)
        Array containing the set of points in space
    mask : array_like, shape (nrows, ncols, 3)
        Floating segmentation mask
    coloring : str, {'OBJECTS', 'FLOATING', 'DEPTH'}
        Points coloring type.
         - FLOATING: each point p is colored with RGB = (255, 0, 0) if p is part of floating object in mask or with the original color in img.
         - DEPTH: each point (x,y,z) is colored with colormap 'jet_r' taking depth[x,y,z] value.
    img : array_like, shape (rows,cols,3), optional
        Original image. If coloring = 'FLOATING', img can't be None.
    
    """

    #nomralize depth values
    # depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

    assert mask is not None, "mask is empty, try to use another mask."
    if coloring == 'FLOATING':
        assert img is not None, "img can't be None if coloring = 'FLOATING"

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
            c=colors
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
            vmin=np.amin(depth),
            vmax = np.amax(depth)
        )
        cmap = mpl.cm.jet_r
        norm = mpl.colors.Normalize(vmin=np.amin(depth), vmax=np.amax(depth))

        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),label='depth estimation value')
    fig.view_init(15,223)

    fig.set_xlim3d(0, ncols)
    fig.set_ylim3d(np.amin(depth), np.amax(depth))
    fig.set_zlim3d(0, nrows)

    fig.set_xlabel(" Y ")
    fig.set_ylabel(" Z ")
    fig.set_zlabel(" X ")

    fig.invert_zaxis()
    if output_path:
        plt.savefig(output_path)
    plt.show(block=False)
    plt.close('all')

def showFinalSegmentation(three_mask, color_mask=None, output_paths=[None, None, None]):
    """Shows both two segmentation images.

    Parameters
    ----------
    three_mask : array_like, shape (nrows, ncols, 3)
        Three class segmentation mask (water, scene, floating objects)
    color_mask : array_like, shape (nrows, ncols, 3), optional
        Object segmentation mask. If color_mask is None, object segmentation is not shown

    """

    # Resize mask
    nrows, ncols,_ = three_mask.shape
    threeMask_resized = cv2.resize(three_mask, (ncols,nrows), interpolation = cv2.INTER_AREA)



    if color_mask is None:
        ncols_plot = 1

    else:
        colorMask_resized = cv2.resize(color_mask, (ncols,nrows), interpolation = cv2.INTER_AREA)
        ncols_plot = 2

    legend_data = [
        [127,list(LIGHT_PURPLE),"water"],
        [126,list(DARK_BLUE),"scene"],
        [125,list(GREEN),"floating"]
    ]
    handles = [
        Rectangle((0,0),1,1, color = [v/255 for v in c]) for k,c,n in legend_data
    ]
    labels = [n for k,c,n in legend_data]


    # # Plot segmentation
    # fig = plt.figure(figsize=plt.figaspect(0.5))
        
    # # ===========================
    # # First subplot (binary mask with floating objects)
    # # ===========================
    # # set up the axes for the first plot
    # ax = fig.add_subplot(1, ncols_plot, 1)
    # ax.set_title('Three class segmentation')
    # ax.imshow(threeMask_resized)

    # ax.grid(False)
    # ax.axis('off')
    # ax.legend(handles,labels)

    # if color_mask is not None:

    #     # ===========================
    #     # Second subplot (color mask)
    #     # ===========================
    #     # set up the axes for the second plot
    #     ax = fig.add_subplot(1, ncols_plot, 2)
    #     ax.set_title('Object segmentation')
    #     ax.imshow(colorMask_resized)

    #     ax.grid(False)
    #     ax.axis('off')
    #     ax.legend([handles[0],handles[2]],[labels[0],labels[2]])
    
    # plt.show()

    # First plot (binary mask with floating objects)
    plt.figure(figsize=plt.figaspect(0.5))
    plt.title('Three class segmentation')
    plt.imshow(threeMask_resized)
    plt.grid(False)
    plt.axis('off')
    # plt.legend(handles,labels)
    if output_paths[0] is not None:
        plt.savefig(output_paths[0])
    plt.show()
    plt.close('all')

    if color_mask is not None:
        # Second plot (color mask)
        plt.figure(figsize=plt.figaspect(0.5))
        plt.title('Object segmentation')
        plt.imshow(colorMask_resized)
        plt.grid(False)
        plt.axis('off')
        # plt.legend([handles[0],handles[2]],[labels[0],labels[2]])
        if output_paths[1] is not None:
            plt.savefig(output_paths[1])
        plt.show()
        plt.close('all')

def segmentationFinal(image_path,coloring,output_paths=[None,None,None]):
    """Get the segmentation combining SAM with depth estimation. If possible, the function returns an object segmentation

    Parameters
    ----------
    image_path : str
        Image path
    coloring : str, {'OBJECTS', 'FLOATING', 'DEPTH'}
        Points coloring type.
         - FLOATING: each point p is colored with RGB = (255, 0, 0) if p is part of floating objetc in mask or with the original color in img.
         - DEPTH: each point (x,y,z) is colored with colormap 'jet_r' taking depth[x,y,z] value.

    Returns
    -------
    binary_mask : array_like
        Image with binary segmentation. If binary mask calculated with SAM does not segment water properly, depth based segmentation will be used.
    color_mask : array_like
        Image with object segmentation. If binary mask calculated with SAM does not segment water properly, color_mask will be None
    """


    def getIntersectAndUnion(merged, segment_index, thresh):
        """Calculates intersection and union values between 'merged' and 'thresh'"""

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
    showBinarySegmentationDepth(image_path,output_paths[0])
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
    while (intersection/union) < 0.7:
        # "Intersect over union" value is not good enough. Search for another water segment
        del new_areas[water_segment_index]
        if len(new_areas) == 0:
            print('Intersection over union is not good enough for any mask. Segmentation based on depth estimation will be used')
            end = time.time()
            print('Final segmentation execution time:',end-start)

            if coloring == 'FLOATING':
                thresh = thresh.astype('uint8')
                floating_mask = floatingSegmentation(thresh)
                showPointcloudWithMask(depth,floating_mask,coloring,img)
                showFinalSegmentation(floating_mask)
            else:
                thresh = thresh.astype('uint8')
                floating_mask = thresh
                showPointcloudWithMask(depth,thresh,coloring,img)
                showFinalSegmentation(floating_mask)
            return thresh, floating_mask, None
        water_segment_index = np.argmax(np.array(new_areas))
        
        intersection, union = getIntersectAndUnion(merged, water_segment_index, thresh)
        
    union += (water - intersection)

    color_mask = img.copy()
    binary_mask = img.copy()
    

    def generate_color():
        while True:
            # Generate a random color in HSV
            h, s, v = np.random.random(), np.random.uniform(0.1, 0.5), np.random.uniform(0.7, 0.95)
            # Convert the HSV color to RGB
            color = [int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v)]
            # Check if the color is not green or light purple
            if not (30 < h*360 < 200 or 220 < h*360 < 330):
                return color

    colors = []
    for i in range(len(masks)):
        for j in range(3):
            colors.append(generate_color())
    
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

    success = True
    if water_percent_SAM/water_percent < 0.7:
        print('No se ha encontrado una máscara binaria mejor. Se utiliza la calculada a partir de la estimación de profundidad')
        success = False
    
    end = time.time()
    # print('Final segmentation execution time:',end-start)

    if success == False:
        thresh = thresh.astype('uint8')
        binary_mask = thresh.copy()
        floating_mask = floatingSegmentation(binary_mask)
        color_mask = None
        showFinalSegmentation(floating_mask,color_mask,output_paths)
    else:
        floating_mask = floatingSegmentation(binary_mask)
        # Paint as green the floating objects in color_mask
        for i in range(merged.shape[0]):
            for j in range(merged.shape[1]):
                if floating_mask[i,j,:].tolist() == list(GREEN):
                    color_mask[i,j] = GREEN

        showFinalSegmentation(floating_mask,color_mask,output_paths)

    if coloring == 'FLOATING':
        showPointcloudWithMask(depth,floating_mask,coloring,img,output_paths[2])
    else:
        showPointcloudWithMask(depth,binary_mask,coloring,img,output_paths[2])

    
    return binary_mask, floating_mask, color_mask

