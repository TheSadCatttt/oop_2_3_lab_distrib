#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

// Âñïîìîãàòåëüíûå ôóíêöèè
double erfcx(double x);
double pdf_sen_standard(double x, double v);

// ÎÑÍÎÂÍÎÅ
double pdf_main(double x, double mu, double lambda, double v);
void moments_main(double mu, double lambda, double v, double* mean,
    double* variance, double* skewness, double* kurtosis);
double generate_main(double mu, double lambda, double v);

// ÑÌÅÑÜ
typedef struct {
    double mu1, lambda1, v1;
    double mu2, lambda2, v2;
    double p;
} MixtureParams;

double pdf_mixture(double x, MixtureParams* params);
void moments_mixture(MixtureParams* params, double* mean, double* variance,
    double* skewness, double* kurtosis);
double generate_mixture(MixtureParams* params);

// İÌÏÈĞÈ×ÅÑÊÎÅ 
typedef struct {
    double* data;
    int n;
    double* frequencies;
    int k;
    double min_val, max_val;
} EmpiricalParams;

void init_empirical(EmpiricalParams* params, double* data, int n, int k);
double pdf_empirical(double x, EmpiricalParams* params);
void moments_empirical(EmpiricalParams* params, double* mean, double* variance,
    double* skewness, double* kurtosis);
double generate_empirical(EmpiricalParams* params);
void free_empirical(EmpiricalParams* params);

#endif