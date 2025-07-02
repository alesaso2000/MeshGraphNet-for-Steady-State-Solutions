
def save_U(internal_values, out_extruded_values, output_file):
    with open(output_file, 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*
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
    location    "1001";
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
        f.write("""/*--------------------------------*- C++ -*----------------------------------*
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
    location    "1001";
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   nonuniform List<scalar> """)
        f.write(f'\n{len(internal_values)}\n')
        f.write('(\n')
        for value in internal_values:
            f.write(f'{value.item()}\n')
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













