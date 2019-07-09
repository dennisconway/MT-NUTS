functions {
  vector c_mult(vector a, vector b) {
    vector[2] c;
    c[1] = a[1]*b[1]-a[2]*b[2];
    c[2] = a[1]*b[2]+a[2]*b[1];
    return c;
  }
  vector c_div(vector a, vector b) {
    vector[2] c;
    c[1] = (a[1]*b[1]+a[2]*b[2])/(square(b[1])+square(b[2]));
    c[2] = (a[2]*b[1]-a[1]*b[2])/(square(b[1])+square(b[2]));
    return c;
  }
  vector c_tanh(vector a) {
    vector[2] c;
    vector[2] denom;
    vector[2] numer;
    denom[1] = 1;
    denom[2] = tanh(a[1])*tan(a[2]);
    numer[1] = tanh(a[1]);
    numer[2] = tan(a[2]);
    c = c_div(numer,denom);
    return c;
  }
  vector z_synth(int L, real freq, vector R, vector T) {
    vector[2] Z;
    vector[2] qn;
    vector[2] C;
    vector[2] resphs;
    vector[2] old_c;
    vector[2] unit_45;
    vector[2] unit_90;
    vector[2] unit_0;
    vector[L] res;
    real mu0;
    real t;
    real omega;
    unit_45[1] = 1/sqrt2();
    unit_45[2] = 1/sqrt2();
    unit_90[1] = 0;
    unit_90[2] = 1;
    unit_0[1] = 1;
    unit_0[2] = 0;
    mu0 = 4*pi()*10^-7;
    omega = 2*pi()*freq;
    qn = sqrt(mu0*omega/10^(R[1]))*unit_45;
    C = c_div(unit_0,qn);
    for (i in 2:L) {
        old_c = C;
        qn = sqrt(mu0*omega/10^(R[i]))*unit_45;
        C = c_div(c_mult(old_c,qn)+c_tanh(qn*T[i-1]),unit_0+c_mult(c_mult(old_c,qn),c_tanh(qn*T[i-1])));
        C = c_div(C,qn);
    }
    Z = c_mult(C,unit_90)*omega/1000;
    return Z;
  }
}
data {
    int<lower=2> L; // number of layers
    int<lower=1> F; // number of frequencies
    real FREQ[F]; // frequency values
    matrix[F,2] z_o; // measured impedance values
    vector[F] sigma; //
    // real
}
parameters {
    // real<lower=1,upper=7> halfspace; // bottom layer resistivity
    vector[L] R; // layer resistivities
    vector<lower=10,upper=1500>[L-1] T; // layer thicknesses
    vector<lower=0>[L-1] a; // smoothness
    //vector<lower=0>[F] b; // errors

    //vector<lower=0>[L-1] Re; //rayleight vars vars
}
transformed parameters {
  vector[2] Z; // temporary impedance values
  matrix[F,2] z_s; // synthetic impedance
      for (i in 1:F) {
        Z = z_synth(L, FREQ[i], R, T);
        z_s[i,1] = Z[1];
        z_s[i,2] = Z[2];
  }
}

model {
      for (i in 1:F) {
        // b[i] ~ exponential(1/sigma[i]);
        // b[i,2] ~ exponential(1/sigma[i]);
        z_o[i,1] ~ normal(z_s[i,1],sigma[i]);
        z_o[i,2] ~ normal(z_s[i,2],sigma[i]);
      }
      for (i in 2:L) {
          R[i] ~ normal(R[i-1],a[i-1]);
          //T[i-1] ~ normal(Re[i-1],Re[i-1]*0.2);
          a[i-1] ~ exponential(0.5);
        //   T[i-1] ~ rayleigh(300*1.1^(L-i));
          //sig[i-1] ~ exponential(300);
      }
      //a[1] ~ exponential(0.5);
      //a[L-1] ~ exponential(0.5);
    }
