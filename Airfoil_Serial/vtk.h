#ifndef VTK_H
#define VTK_H

void set_default_base();
void set_basename(const char *base);
char *get_basename();
int write_checkpoint(int iters, double t);
int write_result(int iters, double t);

#endif
