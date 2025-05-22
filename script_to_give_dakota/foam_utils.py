import Ofpp



def create_boundary_field(openfoam_file, new_data_array, verbose=False):
    data = Ofpp.parse_boundary_field(openfoam_file)
    l = 0
    for patch_name, patch_data in data.items():
        if b'value' in patch_data:
            l_now = len(patch_data[b'value'])
            data[patch_name][b'value'] = new_data_array[l:l + l_now:, :]
            l = l_now
        else:
            if verbose:
                print(f"Warning: b'value' not found for patch '{patch_name}'.")
    return data



# def save_foam_field(ofpp_internal, ofpp_boundary, object_field, output_file, field_class='volVectorField', field_location=str(0), field_dimensions='[0 1 0 0 0 0 0]'):
#     with open(output_file, 'w') as f:
#         f.write('/*--------------------------------*- C++ -*----------------------------------*\\\n')
#         f.write('  =========                 |\n')
#         f.write('  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n')
#         f.write('   \\\\    /   O peration     | Website:  https://openfoam.org\n')
#         f.write('    \\\\  /    A nd           | Version:  *\n')
#         f.write('     \\\\/     M anipulation  |\n')
#         f.write('\\*---------------------------------------------------------------------------*/\n')
#         f.write('FoamFile\n{\n')
#         f.write('    format      ascii;\n')
#         f.write(f"    class       {field_class};\n")
#         f.write(f'    location    "{field_location}";\n')
#         f.write(f"    object      {object_field};\n")
#         f.write('}\n')
#         f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n')

#         f.write(f"dimensions      {field_dimensions};\n\n")

#         # Internal
#         if field_class == 'volVectorField':
#             f.write('internalField   nonuniform List<vector>\n')
#             f.write(f'{len(ofpp_internal)}\n')
#             f.write('(\n')
#             for vector in ofpp_internal:
#                 f.write(f'({vector[0]} {vector[1]} {vector[2]})\n')
#         elif field_class == 'volScalarField':
#             f.write('internalField   nonuniform List<scalar>\n')
#             f.write(f'{len(ofpp_internal)}\n')
#             f.write('(\n')
#             for vector in ofpp_internal:
#                 f.write(f'{vector[0]}\n')
#         f.write(')\n;\n\n')

#         # Boundary
#         f.write('boundaryField\n{\n')
#         for patch_name, patch_data in ofpp_boundary.items():
#             if isinstance(patch_name, bytes):
#                 patch_name = patch_name.decode('utf-8')
#             f.write(f'    {patch_name}\n')
#             f.write('    {\n')
#             if field_class == 'volVectorField':
#                 if b'value' in patch_data:
#                     patch_type = patch_data.get('type', 'calculated')
#                     f.write(f'        type            {patch_type};\n')
#                     field_data = patch_data[b'value']
#                     f.write('        value           nonuniform List<vector>\n')
#                     f.write(f'{len(field_data)}\n')
#                     f.write('(\n')
#                     for vector in field_data:
#                         f.write(f'({vector[0]} {vector[1]} {vector[2]})\n')
#                     f.write(')\n;\n')
#                 else:
#                     f.write(f'        type            empty;\n')
#             elif field_class == 'volScalarField':
#                 if b'value' in patch_data:
#                     patch_type = patch_data.get('type', 'calculated')
#                     f.write(f'        type            {patch_type};\n')

#                     field_data = patch_data[b'value']
#                     f.write('        value           nonuniform List<scalar>\n')
#                     f.write(f'{len(field_data)}\n')
#                     f.write('(\n')
#                     for vector in field_data:
#                         f.write(f'{vector[0]}\n')
#                     f.write(')\n;\n')
#                 else:
#                     f.write(f'        type            empty;\n')
#             f.write('    }\n')
#         f.write('}\n\n\n')
#         f.write('// ************************************************************************* //\n')


def save_U(internal_values, out_extruded_values, output_file):
    with open(output_file, 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volVectorField;
    location    "1000";
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector> """)
        f.write(f'\n{len(internal_values)}\n')
        f.write('(\n')
        for vector in internal_values:
            f.write(f'({vector[0]} {vector[1]} {vector[2]})\n')

        # Boundary
        f.write(""")
;

boundaryField
{
    prof_extruded
    {
        type            fixedValue;
        value           uniform (0 0 0);
    }
    free_extruded
    {
        type            slip;
    }
    in_extruded
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }
    out_extruded
    {
        type            inletOutlet;
        inletValue      uniform (0 0 0);
        value           nonuniform List<vector>""")

        f.write(f"\n{len(out_extruded_values)}\n")
        f.write('(\n')
        for vec in out_extruded_values:
            f.write(f"({vec[0]} {vec[1]} {vec[2]})\n")
        f.write(""")
    ;
        }
        defaultFaces
        {
            type            empty;
        }
    }


    // ************************************************************************* //""")
        


def save_pressure(internal_values, output_file):
    with open(output_file, 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volScalarField;
    location    "1000";
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   nonuniform List<scalar> """)
        f.write(f'\n{len(internal_values)}\n')
        f.write('(\n')
        for value in internal_values:
            f.write(f'{value}\n')
        f.write(""")
;

boundaryField
{
    prof_extruded
    {
        type            zeroGradient;
    }
    free_extruded
    {
        type            slip;
    }
    in_extruded
    {
        type            zeroGradient;
    }
    out_extruded
    {
        type            fixedValue;
        value           uniform 0;
    }
    defaultFaces
    {
        type            empty;
    }
}


// ************************************************************************* //""")














