#include <stdio.h>
#include <stdlib.h>

int main ()
{
    for (int i = 2; i <= 127; i++)
    {
        FILE* size_file = fopen ("size.ini", "w");
	if (!size_file) return 1;
	fprintf (size_file, "%d", i);
	fclose (size_file);
	system ("make r");
    }

    return 0;
}
