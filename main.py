from analyzer import FloorPlanAnalyzer


def main():
    """Main script to run the floor plan analyzer."""
    
    # Initialize analyzer with default calibration
    analyzer = FloorPlanAnalyzer(pixels_per_meter=50.0)
    
    # Hardcoded input file
    input_file = "plan.pdf"
    
    try:
        # Analyze the floor plan
        print(f"Analyzing floor plan: {input_file}")
        wall_length, processed_image = analyzer.analyze(input_file)
        
        print(f"\nAnalysis Results:")
        print(f"Total wall length: {wall_length:.2f} meters")
        
        # Export BoQ to CSV
        print(f"\nExporting Bill of Quantities to boq.csv...")
        boq_df = analyzer.export_boq(wall_length)
        
        print(f"\nBoQ Contents:")
        print(boq_df.to_string(index=False))
        print(f"\nBoQ exported successfully to boq.csv")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

