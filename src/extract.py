import os
import argparse

from tqdm import tqdm
from PIL import Image
import geopandas as gp
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.ops import transform as shapely_transform
import rasterio
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import mapping

from src.tiles import tiles_from_slippy_map
from src.features.building import Roof_features

def mask_to_features_multiclass(mask_dir):
    """
    Convert masks to separate features for each class.
    Returns a dictionary with class indices as keys.
    """
    handlers = {i: Roof_features() for i in range(1, 5)}  # Classes 1-4
    tiles = list(tiles_from_slippy_map(mask_dir))

    for tile, path in tqdm(tiles, ascii=True, unit="mask"):
        image = np.array(Image.open(path).convert("P"), dtype=np.uint8)
        
        # Process each class separately
        for class_idx in range(1, 5):
            mask = (image == class_idx).astype(np.uint8)
            handlers[class_idx].apply(tile, mask)
    
    # Output feature collections for each class
    features = {i: handlers[i].jsonify() for i in range(1, 5)}
    
    return features


def assign_priority_class(labels):
    """
    Assign class based on priority rules:
    Priority order: 3 > 2 > 1 > 4
    
    Rules:
    - If label 3 is present (with any other labels), choose 3
    - If label 2 is present (with any other labels except 3), choose 2
    - If labels 1 and 2, choose 2
    - If label 4 is present with any other label, choose the other label
    - If only one label, return that label
    
    Args:
        labels: list or set of class labels
    
    Returns:
        int: The priority class
    """
    labels = set(labels)
    
    # Priority 1: If 3 is present, always choose 3
    if 3 in labels:
        return 3
    
    # Priority 2: If 2 is present, choose 2
    if 2 in labels:
        return 2
    
    # Priority 3: If 1 is present (without 2 or 3), choose 1
    if 1 in labels:
        return 1
    
    # Priority 4: If only 4 remains, choose 4
    if 4 in labels:
        return 4
    
    # Fallback (shouldn't reach here with classes 1-4)
    return min(labels)


def assign_by_max_area(area_percentages, slope_threshold=6.5, slope_value=None):
    """
    Assign class based on which label has the maximum area percentage.
    Only considers classes if slope >= slope_threshold.
    
    Args:
        area_percentages: dict with class_id as keys and percentage as values
        slope_threshold: minimum slope value to consider (default 6.5)
        slope_value: actual slope value for the building
    
    Returns:
        int: The class with maximum area (or None if slope < threshold)
    """
    if not area_percentages:
        return None
    
    # If slope data is available and slope is below threshold, return None
    if slope_value is not None and slope_value < slope_threshold:
        return None
    
    # Return class with maximum area percentage
    valid_percentages = {k: v for k, v in area_percentages.items() if v > 0}
    if not valid_percentages:
        return None
    
    return max(valid_percentages.items(), key=lambda x: x[1])[0]


def extract_slope_from_raster(buildings_gdf, dtm_path):
    """
    Extract slope values from DTM raster for each building polygon.
    
    Args:
        buildings_gdf: GeoDataFrame with building polygons
        dtm_path: Path to DTM slope raster (GeoTIFF)
    
    Returns:
        GeoDataFrame with added 'mean_slope' column
    """
    try:
        print(f"\n  Opening DTM raster: {dtm_path}")
        
        with rasterio.open(dtm_path) as src:
            print(f"  ✓ DTM CRS: {src.crs}")
            print(f"  ✓ DTM bounds: {src.bounds}")
            print(f"  ✓ DTM shape: {src.shape}")
            print(f"  ✓ DTM resolution: {src.res}")
            
            # Convert buildings to same CRS as raster if needed
            buildings_transformed = buildings_gdf.copy()
            if buildings_gdf.crs != src.crs:
                print(f"  ⚠ Converting buildings from {buildings_gdf.crs} to {src.crs}")
                buildings_transformed = buildings_transformed.to_crs(src.crs)
            
            # Extract slope values for each building
            slope_values = []
            
            print(f"  Extracting slope values for {len(buildings_transformed)} buildings...")
            for idx, row in tqdm(buildings_transformed.iterrows(), total=len(buildings_transformed), desc="  Processing buildings"):
                try:
                    # Get the geometry
                    geom = row['geometry']
                    
                    # Mask the raster with the building polygon
                    out_image, out_transform = rasterio_mask(src, [mapping(geom)], crop=True, nodata=np.nan)
                    
                    # Extract values (first band)
                    values = out_image[0]
                    
                    # Calculate mean slope (excluding nodata values)
                    valid_values = values[~np.isnan(values)]
                    
                    if len(valid_values) > 0:
                        mean_slope = np.mean(valid_values)
                    else:
                        mean_slope = 0.0
                    
                    slope_values.append(mean_slope)
                    
                except Exception as e:
                    # If extraction fails for a building, assign 0
                    slope_values.append(0.0)
            
            # Add slope values to original GeoDataFrame
            buildings_gdf['mean_slope'] = slope_values
            
            print(f"  ✓ Slope extraction complete")
            print(f"  ✓ Slope range: {min(slope_values):.2f}° to {max(slope_values):.2f}°")
            print(f"  ✓ Mean slope: {np.mean(slope_values):.2f}°")
            
            return buildings_gdf
            
    except Exception as e:
        print(f"  ✗ Error reading DTM raster: {e}")
        buildings_gdf['mean_slope'] = 0.0
        return buildings_gdf


def intersection(target_type, city_name, mask_dir, use_area_based=True, dtm_path=None, slope_threshold=6.5):
    """
    Process all 4 classes and combine them into a single GeoJSON with class labels.
    
    Args:
        target_type: Base name for output files (e.g., 'Green')
        city_name: Name of the city
        mask_dir: Directory containing masks
        use_area_based: If True, use area-based assignment; if False, use priority-based
        dtm_path: Path to DTM slope raster (GeoTIFF) (optional)
        slope_threshold: Minimum slope value to assign labels (default 6.5)
    """
    # Define class names - customize these based on your classes
    class_names = {
        1: 'Green',
        2: 'Potential_Green', 
        3: 'Solar',
        4: 'Flat_but_Not_Possible_to_Green'
    }
    
    print()
    print("=" * 80)
    print("STAGE 1: CONVERTING PREDICTION MASKS TO GEOJSON FEATURES")
    print("=" * 80)
    features_dict = mask_to_features_multiclass(mask_dir)
    
    # Report Stage 1: Labels before intersection
    print("\n📊 STAGE 1 RESULTS - Model Predictions (Before Intersection):")
    print("-" * 80)
    total_predictions = 0
    for class_idx, features in features_dict.items():
        count = len(features['features'])
        total_predictions += count
        class_name = class_names.get(class_idx, f'Class{class_idx}')
        print(f"  Class {class_idx} ({class_name}): {count} predicted features")
    print(f"\n  Total predicted features across all classes: {total_predictions}")
    print("-" * 80)
    
    # Save Stage 1 predictions to files
    output_dir = 'results/04Results'
    stage1_dir = os.path.join(output_dir, 'stage1_predictions')
    os.makedirs(stage1_dir, exist_ok=True)
    
    print(f"\n💾 Saving Stage 1 predictions to: {stage1_dir}")
    all_predictions = []
    
    for class_idx, features in features_dict.items():
        if len(features['features']) > 0:
            class_name = class_names.get(class_idx, f'Class{class_idx}')
            
            # Convert to GeoDataFrame
            gdf = gp.GeoDataFrame.from_features(features, crs=4326)
            gdf['class_id'] = class_idx
            gdf['class_name'] = class_name
            
            # Save individual class file
            class_output = os.path.join(stage1_dir, f'{city_name}_class_{class_idx}_{class_name}.geojson')
            gdf.to_file(class_output, driver='GeoJSON')
            print(f"  ✓ Saved Class {class_idx} ({class_name}): {class_output}")
            
            all_predictions.append(gdf)
    
    # Save combined predictions (all classes together)
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        combined_predictions = gp.GeoDataFrame(combined_predictions, crs=4326)
        
        combined_output = os.path.join(stage1_dir, f'{city_name}_all_predictions_combined.geojson')
        combined_predictions.to_file(combined_output, driver='GeoJSON')
        print(f"\n  ✓ Saved combined predictions: {combined_output}")
    
    # loading building polygons
    city = 'results/01City/' + city_name + '.geojson'
    city = gp.GeoDataFrame.from_file(city)[['geometry']]
    print(f"\n  Total building polygons loaded: {len(city)}")
    
    # Extract DTM slope values if provided
    if dtm_path:
        print()
        print("=" * 80)
        print("LOADING DTM SLOPE DATA FROM RASTER")
        print("=" * 80)
        
        if os.path.exists(dtm_path):
            city = extract_slope_from_raster(city, dtm_path)
            
            # Report slope statistics
            buildings_above_threshold = (city['mean_slope'] >= slope_threshold).sum()
            buildings_below_threshold = (city['mean_slope'] < slope_threshold).sum()
            print(f"\n  ✓ Slope threshold: {slope_threshold}°")
            print(f"  ✓ Buildings with slope >= {slope_threshold}°: {buildings_above_threshold}")
            print(f"  ✓ Buildings with slope < {slope_threshold}°: {buildings_below_threshold}")
            
            # FILTER: Keep only buildings with slope < threshold
            print(f"\n  ⚠ Filtering: Keeping only buildings with slope < {slope_threshold}°")
            city = city[city['mean_slope'] < slope_threshold]
            print(f"  ✓ Buildings after filtering: {len(city)}")
        else:
            print(f"  ✗ DTM file not found: {dtm_path}")
            city['mean_slope'] = None
    else:
        city['mean_slope'] = None
    
    # Create transformer
    transformer = Transformer.from_crs(city.crs, 'epsg:3395', always_xy=True)

    def transform_and_area(geom):
        """Transform geometry and calculate area"""
        if geom is None or geom.is_empty:
            return 0
        try:
            geom_projected = shapely_transform(transformer.transform, geom)
            return geom_projected.area
        except Exception as e:
            print(f"Error transforming geometry: {e}")
            return 0

    # Apply transformation
    city['area'] = city['geometry'].apply(transform_and_area)
    
    # Create a unique identifier based on geometry
    city['geom_wkt'] = city['geometry'].apply(lambda g: g.wkt)
    
    print()
    print("=" * 80)
    print("STAGE 2: INTERSECTING PREDICTIONS WITH BUILDING FOOTPRINTS")
    print("=" * 80)
    
    
    all_intersections = []
    stage2_stats = {}

    for class_idx, features in features_dict.items():
        if len(features['features']) == 0:
            print(f"\n  Class {class_idx}: No features to intersect")
            continue
            
        prediction = gp.GeoDataFrame.from_features(features, crs=4326) 
        
        # Calculate area of prediction features
        prediction['pred_area'] = prediction.geometry.apply(transform_and_area)
        
        # Perform spatial join
        intersections = gp.sjoin(city, prediction, how="inner", predicate='intersects')
        
        # Calculate intersection areas using vectorized operations
        # Get the original geometries before they were modified by sjoin
        city_geoms = intersections['geometry'].values
        pred_indices = intersections['index_right'].values
        pred_geoms = prediction.loc[pred_indices, 'geometry'].values
        
        # Calculate intersections
        intersection_geoms = [city_geoms[i].intersection(pred_geoms[i]) 
                            for i in range(len(city_geoms))]
        intersection_areas = [transform_and_area(geom) for geom in intersection_geoms]
        
        intersections['intersection_area'] = intersection_areas
        
        # Add class information
        class_name = class_names.get(class_idx, f'Class{class_idx}')
        intersections['class_id'] = class_idx
        intersections['class_name'] = class_name
        
        all_intersections.append(intersections)
        stage2_stats[class_idx] = len(intersections)
        print(f"\n  Class {class_idx} ({class_name}):")
        print(f"    - Predicted features: {len(features['features'])}")
        print(f"    - Building intersections: {len(intersections)}")
    
    # Report Stage 2: Labels after intersection (without priority)
    print("\n📊 STAGE 2 RESULTS - After Intersection (Before Assignment):")
    print("-" * 80)
    total_intersections = sum(stage2_stats.values())
    for class_idx, count in stage2_stats.items():
        class_name = class_names.get(class_idx, f'Class{class_idx}')
        print(f"  Class {class_idx} ({class_name}): {count} building intersections")
    print(f"\n  Total intersections (may include duplicates): {total_intersections}")
    print("-" * 80)
    
    # Combine all classes into one GeoDataFrame
    if all_intersections:
        combined = pd.concat(all_intersections, ignore_index=True)
        
        # Check for buildings with multiple labels
        unique_buildings = combined['geom_wkt'].nunique()
        print(f"\n  Unique buildings with intersections: {unique_buildings}")
        print(f"  Buildings appearing multiple times: {total_intersections - unique_buildings}")
        
        print()
        print("=" * 80)
        print("STAGE 3: CALCULATING AREA PERCENTAGES AND ASSIGNING FINAL LABELS")
        print("=" * 80)
        
        if dtm_path and city['mean_slope'].notna().any():
            print(f"\n⚠ Slope Filtering: Only buildings with slope >= {slope_threshold}° will be assigned labels")
        
        if use_area_based:
            print("Assignment Method: Area-based (label with maximum area percentage)")
        else:
            print("\nPriority Rules:")
            print("  1. Label 3 (Solar) > all others")
            print("  2. Label 2 (Potential_Green) > Label 1, 4")
            print("  3. Label 1 (Green) > Label 4")
            print("  4. Label 4 (Flat_but_Not_Possible_to_Green) = lowest priority")
        print()
        
        # Group by geometry (using WKT string representation)
        building_groups = combined.groupby('geom_wkt')
        
        final_rows = []
        assignment_changes = []
        multi_label_buildings = []
        filtered_by_slope = 0
        
        for geom_wkt, group in building_groups:
            # Get building total area
            building_area = group['area'].iloc[0]
            
            # Get slope value
            slope_value = group['mean_slope'].iloc[0] if 'mean_slope' in group.columns else None
            
            # Calculate area percentage for each class
            area_percentages = {}
            for class_idx in range(1, 5):
                class_group = group[group['class_id'] == class_idx]
                if len(class_group) > 0:
                    # Sum intersection areas for this class
                    total_intersection_area = class_group['intersection_area'].sum()
                    percentage = (total_intersection_area / building_area * 100) if building_area > 0 else 0
                    area_percentages[class_idx] = percentage
                else:
                    area_percentages[class_idx] = 0.0
            
            # Get all class labels for this building
            labels = sorted(group['class_id'].unique().tolist())
            
            # Check slope threshold
            passes_slope_filter = True
            if slope_value is not None and not np.isnan(slope_value) and slope_value < slope_threshold:
                passes_slope_filter = False
                filtered_by_slope += 1
            
            # Assign final class
            if passes_slope_filter:
                if use_area_based:
                    # Area-based: choose class with maximum area percentage
                    final_class = assign_by_max_area({k: v for k, v in area_percentages.items() if v > 0})
                else:
                    # Priority-based
                    final_class = assign_priority_class(labels)
            else:
                final_class = None  # Building filtered out by slope
            
            if len(labels) > 1:
                # Building has multiple labels
                original_labels = [class_names[l] for l in labels]
                final_class_name = class_names[final_class] if final_class is not None else 'Filtered (slope < threshold)'
                
                multi_label_buildings.append({
                    'original_labels': labels,
                    'original_names': original_labels,
                    'final_class': final_class,
                    'final_name': final_class_name,
                    'area_percentages': area_percentages,
                    'slope': slope_value
                })
                
                assignment_changes.append({
                    'from': ', '.join(map(str, labels)),
                    'to': final_class if final_class is not None else 'None',
                    'from_names': ' + '.join(original_labels),
                    'to_name': final_class_name,
                    'area_percentages': area_percentages,
                    'slope': slope_value
                })
            
            # Get the row with the final class (take first if multiple, or first available if final_class is None)
            if final_class is not None:
                final_row = group[group['class_id'] == final_class].iloc[0].copy()
            else:
                final_row = group.iloc[0].copy()
            
            # Add area percentage columns
            final_row['area_pct_class_1'] = area_percentages[1]
            final_row['area_pct_class_2'] = area_percentages[2]
            final_row['area_pct_class_3'] = area_percentages[3]
            final_row['area_pct_class_4'] = area_percentages[4]
            final_row['final_class_id'] = final_class if final_class is not None else -1
            final_row['final_class_name'] = class_names[final_class] if final_class is not None else 'Filtered_by_slope'
            final_row['passes_slope_filter'] = passes_slope_filter
            
            # Calculate total predicted area coverage
            total_predicted_pct = sum(area_percentages.values())
            final_row['total_predicted_pct'] = total_predicted_pct
            
            final_rows.append(final_row)
        
        # Create final GeoDataFrame
        result = gp.GeoDataFrame(final_rows, crs=4326)
        
        # Remove the temporary geom_wkt column
        if 'geom_wkt' in result.columns:
            result = result.drop(columns=['geom_wkt'])
        
        # Drop unnecessary columns
        result = result.drop(columns=['intersection_area', 'pred_area'], errors='ignore')
        
        # Report Stage 3: Final labels
        print("\n📊 STAGE 3 RESULTS - Final Labels with Area Percentages:")
        print("-" * 80)
        print(f"\n  Total unique buildings in final output: {len(result)}")
        
        if dtm_path and result['mean_slope'].notna().any():
            print(f"  Buildings filtered by slope (< {slope_threshold}°): {filtered_by_slope}")
            print(f"  Buildings passing slope filter: {len(result) - filtered_by_slope}")
        
        print(f"  Buildings with multiple labels: {len(multi_label_buildings)}")
        print(f"  Buildings with single label: {len(result) - len(multi_label_buildings)}")
        
        print("\n  Final class distribution:")
        # Exclude filtered buildings from main distribution
        valid_results = result[result['final_class_id'] != -1]
        for class_id in sorted(valid_results['final_class_id'].unique()):
            count = len(valid_results[valid_results['final_class_id'] == class_id])
            class_name = class_names[class_id]
            percentage = (count / len(valid_results)) * 100 if len(valid_results) > 0 else 0
            print(f"    Class {class_id} ({class_name}): {count} ({percentage:.1f}%)")
        
        if filtered_by_slope > 0:
            percentage = (filtered_by_slope / len(result)) * 100
            print(f"    Filtered by slope: {filtered_by_slope} ({percentage:.1f}%)")
        
        # Show area percentage statistics
        print("\n  Area percentage statistics:")
        for class_id in range(1, 5):
            col_name = f'area_pct_class_{class_id}'
            avg_pct = result[col_name].mean()
            max_pct = result[col_name].max()
            buildings_with_class = (result[col_name] > 0).sum()
            class_name = class_names[class_id]
            print(f"    Class {class_id} ({class_name}):")
            print(f"      - Buildings with this class: {buildings_with_class}")
            print(f"      - Average area %: {avg_pct:.2f}%")
            print(f"      - Maximum area %: {max_pct:.2f}%")
        
        # Show examples of assignments
        if assignment_changes:
            print("\n  Examples of label assignments (showing area percentages and slope):")
            print("  " + "-" * 76)
            
            # Show first 10 examples
            for i, change in enumerate(assignment_changes[:10]):
                if i >= 10:
                    break
                area_str = ", ".join([f"C{k}:{v:.1f}%" for k, v in change['area_percentages'].items() if v > 0])
                slope_str = f"{change['slope']:.2f}°" if change['slope'] is not None and not np.isnan(change['slope']) else "N/A"
                print(f"    [{change['from_names']}] -> [{change['to_name']}]")
                print(f"      Area %: {area_str}, Slope: {slope_str}")
            
            if len(assignment_changes) > 10:
                print(f"    ... and {len(assignment_changes) - 10} more")
        
        print("-" * 80)
        
        # Save outputs
        os.makedirs(output_dir, exist_ok=True)
        
        # Save final result (all buildings including filtered)
        output_path = f'{output_dir}/{city_name}_{target_type}_all_classes_with_areas.geojson'
        result.to_file(output_path, driver='GeoJSON')
        
        # Save only buildings that passed slope filter
        if dtm_path and result['mean_slope'].notna().any():
            valid_results = result[result['passes_slope_filter'] == True]
            valid_output_path = f'{output_dir}/{city_name}_{target_type}_slope_filtered.geojson'
            valid_results.to_file(valid_output_path, driver='GeoJSON')
            print(f"\n💾 Saved slope-filtered results ({len(valid_results)} buildings): {valid_output_path}")
        
        # Save intermediate result (before assignment) for comparison
        combined_before = combined.copy()
        if 'geom_wkt' in combined_before.columns:
            combined_before = combined_before.drop(columns=['geom_wkt'])
        combined_before = gp.GeoDataFrame(combined_before, crs=4326)
        before_assignment_path = f'{output_dir}/{city_name}_{target_type}_before_assignment.geojson'
        combined_before.to_file(before_assignment_path, driver='GeoJSON')
        
        # Save detailed report (with UTF-8 encoding to handle special characters)
        report_path = f'{output_dir}/{city_name}_{target_type}_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"MULTI-CLASS PREDICTION REPORT - {city_name}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Assignment Method: {'Area-based (max area %)' if use_area_based else 'Priority-based'}\n")
            if dtm_path:
                f.write(f"Slope Filter: Buildings with slope >= {slope_threshold}° only\n")
                f.write(f"DTM Source: {dtm_path}\n")
            f.write("\n")
            
            f.write("STAGE 1: Model Predictions (Before Intersection)\n")
            f.write("-" * 80 + "\n")
            for class_idx, features in features_dict.items():
                count = len(features['features'])
                class_name = class_names.get(class_idx, f'Class{class_idx}')
                f.write(f"Class {class_idx} ({class_name}): {count} predicted features\n")
            f.write(f"\nTotal predicted features: {total_predictions}\n\n")
            
            f.write("Files saved in: " + stage1_dir + "\n")
            f.write("  - Individual class GeoJSON files\n")
            f.write("  - Combined predictions GeoJSON\n\n")
            
            f.write("STAGE 2: After Intersection (Before Assignment)\n")
            f.write("-" * 80 + "\n")
            for class_idx, count in stage2_stats.items():
                class_name = class_names.get(class_idx, f'Class{class_idx}')
                f.write(f"Class {class_idx} ({class_name}): {count} intersections\n")
            f.write(f"\nTotal intersections: {total_intersections}\n")
            f.write(f"Unique buildings: {unique_buildings}\n\n")
            
            if dtm_path and result['mean_slope'].notna().any():
                f.write("Slope Filtering:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Buildings filtered by slope (< {slope_threshold}°): {filtered_by_slope}\n")
                f.write(f"Buildings passing slope filter: {len(result) - filtered_by_slope}\n\n")
            
            f.write("STAGE 3: Final Results (With Area Percentages)\n")
            f.write("-" * 80 + "\n")
            valid_results = result[result['final_class_id'] != -1]
            for class_id in sorted(valid_results['final_class_id'].unique()):
                count = len(valid_results[valid_results['final_class_id'] == class_id])
                class_name = class_names[class_id]
                percentage = (count / len(valid_results)) * 100 if len(valid_results) > 0 else 0
                f.write(f"Class {class_id} ({class_name}): {count} ({percentage:.1f}%)\n")
            
            if filtered_by_slope > 0:
                percentage = (filtered_by_slope / len(result)) * 100
                f.write(f"Filtered by slope: {filtered_by_slope} ({percentage:.1f}%)\n")
            
            f.write(f"\nTotal unique buildings: {len(result)}\n")
            f.write(f"Buildings with multiple labels: {len(multi_label_buildings)}\n\n")
            
            f.write("Area Percentage Statistics:\n")
            f.write("-" * 80 + "\n")
            for class_id in range(1, 5):
                col_name = f'area_pct_class_{class_id}'
                avg_pct = result[col_name].mean()
                max_pct = result[col_name].max()
                buildings_with_class = (result[col_name] > 0).sum()
                class_name = class_names[class_id]
                f.write(f"Class {class_id} ({class_name}):\n")
                f.write(f"  Buildings with this class: {buildings_with_class}\n")
                f.write(f"  Average area %: {avg_pct:.2f}%\n")
                f.write(f"  Maximum area %: {max_pct:.2f}%\n\n")
            
            if assignment_changes:
                f.write("Label Assignment Examples (with area percentages and slope):\n")
                f.write("-" * 80 + "\n")
                for change in assignment_changes[:20]:  # Show first 20
                    area_str = ", ".join([f"C{k}:{v:.1f}%" for k, v in change['area_percentages'].items() if v > 0])
                    slope_str = f"{change['slope']:.2f}°" if change['slope'] is not None else "N/A"
                    f.write(f"[{change['from_names']}] -> [{change['to_name']}]\n")
                    f.write(f"  Area %: {area_str}, Slope: {slope_str}\n\n")
        
        print()
        print("=" * 80)
        print("PROCESS COMPLETE!")
        print("=" * 80)
        print(f"\nOutput files saved:")
        print(f"\n  Stage 1 (Raw Predictions):")
        print(f"    - Directory: {stage1_dir}")
        print(f"    - Individual class GeoJSON files")
        print(f"    - Combined predictions GeoJSON")
        print(f"\n  Stage 2 & 3 (Intersection Results):")
        print(f"    1. Final result with areas (all buildings): {output_path}")
        if dtm_path is not None:
            print(f"    2. Slope-filtered result (slope >= {slope_threshold}°): {valid_output_path}")
        print(f"    3. Before assignment: {before_assignment_path}")
        print(f"    4. Detailed report: {report_path}")
        print(f"\n  Final GeoJSON includes these columns:")
        print(f"    - area_pct_class_1: % of building covered by Class 1")
        print(f"    - area_pct_class_2: % of building covered by Class 2")
        print(f"    - area_pct_class_3: % of building covered by Class 3")
        print(f"    - area_pct_class_4: % of building covered by Class 4")
        print(f"    - mean_slope: Average slope of building (if DTM provided)")
        print(f"    - passes_slope_filter: Boolean indicating if slope >= threshold")
        print(f"    - final_class_id: Assigned class ID")
        print(f"    - final_class_name: Assigned class name")
        print(f"    - total_predicted_pct: Total % of building with predictions")
        print()
        
        return result
    else:
        print("\n❌ No features found in any class!")
        return None


def intersection_from_file(prediction_path, target_type, city_name, mask_dir):
    # predicted features
    print()
    print("Loading Prediction Features from file")
    prediction = gp.GeoDataFrame.from_file(prediction_path)[['geometry']]  

    # loading building polygons
    city = 'results/01City/' + city_name + '.geojson'
    city = gp.GeoDataFrame.from_file(city)[['geometry']]  
    city['area'] = city['geometry'].to_crs({'init': 'epsg:3395'}).map(lambda p: p.area)
    
    intersections = gp.sjoin(city, prediction, how="inner", predicate='intersects')
    intersections = intersections.drop_duplicates(subset=['geometry'])
    intersections.to_file('results/04Results/' + city_name + '_' + target_type + ".geojson", driver='GeoJSON')
    
    print()
    print("Process complete, footprints with " + target_type + " roofs are saved at results/04Results/" + city_name + '_' + target_type + ".geojson")
    return intersections


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("city", help="City to be predicted, must be the same as the name of the dataset")
    parser.add_argument("type", help="Roof Typology base name (e.g., Green, Solar)")
    parser.add_argument("--priority", action="store_true", help="Use priority-based assignment instead of area-based")
    parser.add_argument("--dtm", type=str, default="C:/Users/Ko Ko/ISSI/GreenRoof/Swisstopo Dataset for whole study region/swiss topo swissalt3d/mosaic/swisstopo_swissalt3D_bern_mosaic_Slope_2025.tif", help="Path to DTM slope shapefile/geojson")
    parser.add_argument("--slope-column", type=str, default="slope", help="Name of slope column in DTM (default: 'slope')")
    parser.add_argument("--slope-threshold", type=float, default=6.5, help="Minimum slope threshold in degrees (default: 6.5)")
    args = parser.parse_args()

    city_name = args.city
    target_type = args.type
    mask_dir = os.path.join("results", "03Masks", target_type, city_name)
    
    # Default to area-based assignment unless --priority flag is used
    use_area_based = not args.priority
    
   # Pass dtm_path and slope_threshold to intersection function
    intersection(
        target_type, 
        city_name, 
        mask_dir, 
        use_area_based=use_area_based,
        dtm_path=args.dtm,
        slope_threshold=args.slope_threshold
    )