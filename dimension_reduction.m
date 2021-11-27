%Reading the data from the dataset csv file
data=readmatrix("D:\Windsor\Fourth semester\Applied Machine learning\Project\Datasets\Datasets\D3.csv");
CE_Dir="D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D3_CE.csv";
MVU_Dir="D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D3_MVU.csv";
LMVU_Dir="D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D3_LMVU.csv";

%No of output Dimensions 
dim_CE=9;
dim_MVU=9;
dim_LMVU=6;

%Getting the size of the data matrix
[m,n]= size(data);

%fill missing values with median of nieghbours
data_f = fillmissing(data,'constant',0);

%remove columns filled with zeros
data_f = data_f(:,any(data_f));

%Getting the size of the processed data matrix
[m,n]= size(data_f);

iter=round(dim_CE-1/3);
batch_size=round(n/iter)-1;
idx=[];

for i=1:iter
    idx(i)=i*batch_size;
end

%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%
%Conformal Eigenmaps
%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%


transformed_matrix=[];

for i=1:iter
    start_idx=(i-1)*batch_size+1;
    end_idx=idx(i);
    
    %Dimension Reduction using Conformal Eigenmaps
    %For every batch
    [mappedData,mapping]= compute_mapping(data_f(:,start_idx:n-1),"CCA",3,2);
    
    [m_mapped,n_mapped] = size(mappedData);
    [m_transformed,n_transformed] =size(transformed_matrix);
    
    if(m_transformed==0)
        m_transformed=m;
    end
    
    if(m_transformed>m_mapped)
        eigen=zeros(abs(m_transformed-m_mapped),3);
        [m_mapped,n_mapped] = size([mappedData(:,:);eigen]);
        [m_transformed,n_transformed] =size(transformed_matrix);
       
        transformed_matrix=[transformed_matrix(:,:) [mappedData(:,:);eigen]];
    else
        
        eigen=zeros(abs(m_transformed-m_mapped),3*i);
        
        [m_mapped,n_mapped] = size(mappedData);
        [m_eigen,n_eigen]=size(eigen);
        [m_transformed,n_transformed] =size(transformed_matrix);
      
        transformed_matrix=[[transformed_matrix(:,:);eigen] mappedData(1:m_transformed,:)];
    end
    
    max=0;
    label=[];
    if(max<length(mapping.conn_comp))
        max=length(mapping.conn_comp);
        label=mapping.conn_comp;
    end
end

[m_transformed,n_transformed] =size(transformed_matrix);

corres_labels=data_f(label,n);

[m_labels,n_labels] =size(corres_labels);

temp=zeros(abs(m_transformed-m_labels),1);

transformed_matrix(:,n_transformed+1)=[corres_labels;temp];

%Writing the data to a CSV
writematrix(transformed_matrix,CE_Dir,"Delimiter","comma");
disp("The dimension reduced (Conformal Eigenmaps) dataset has been written to a CSV file")

%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%
%Maximum Variance unfolding
%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%

iter=round(dim_MVU-1/3);
batch_size=round(n/iter)-1;
idx=[];

for i=1:iter
    idx(i)=i*batch_size;
end


transformed_matrix=[];

for i=1:iter
    start_idx=(i-1)*batch_size+1;
    end_idx=idx(i);
    
    %Dimension Reduction using Conformal Eigenmaps
    %For every batch
    [mappedData,mapping]= compute_mapping(data_f(:,start_idx:n-1),"MVU",3,2);
    
    [m_mapped,n_mapped] = size(mappedData);
    [m_transformed,n_transformed] =size(transformed_matrix);

    if(m_transformed==0)
        m_transformed=m;
    end
    
    if(m_transformed>m_mapped)
        eigen=zeros(abs(m_transformed-m_mapped),3);
        [m_mapped,n_mapped] = size([mappedData(:,:);eigen]);
        [m_transformed,n_transformed] =size(transformed_matrix);
        
        transformed_matrix=[transformed_matrix(:,:) [mappedData(:,:);eigen]];
    else
        
        eigen=zeros(abs(m_transformed-m_mapped),3*i);
        
        [m_mapped,n_mapped] = size(mappedData);
        [m_eigen,n_eigen]=size(eigen);
        [m_transformed,n_transformed] =size(transformed_matrix);
        
        transformed_matrix=[[transformed_matrix(:,:);eigen] mappedData(1:m_transformed,:)];
    end
    
    max=0;
    label=[];
    if(max<length(mapping.conn_comp))
        max=length(mapping.conn_comp);
        label=mapping.conn_comp;
    end
end

[m_transformed,n_transformed] =size(transformed_matrix);

corres_labels=data_f(label,n);

[m_labels,n_labels] =size(corres_labels);

temp=zeros(abs(m_transformed-m_labels),1);

transformed_matrix(:,n_transformed+1)=[corres_labels;temp];

%Writing the data to a CSV
writematrix(transformed_matrix,MVU_Dir,"Delimiter","comma");
disp("The dimension reduced (Maximum Variance Unfolding) dataset has been written to a CSV file")

%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%
%Landmark Maximum Variance unfolding
%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%

iter=round(dim_LMVU+2/3);
batch_size=round(n/iter)-1;
idx=[];

for i=1:iter
    idx(i)=i*batch_size;
end


TotalOutputRows=1000;
data_f=sortrows(data_f,n);
data_LMVU=[];
labels_unique=unique(data_f(:,n));
noOfLabels=length(labels_unique);
RowsPerLabel=round(TotalOutputRows/noOfLabels);
for i=1:noOfLabels
    temp_list=data_f(data_f(:,n)==labels_unique(i),:);
    [m_temp,n_temp]=size(temp_list);
    [m_LMVU,n_LMVU]=size(data_LMVU);
    if RowsPerLabel<m_temp
        data_LMVU=[data_LMVU;temp_list(1:RowsPerLabel,:)];
    else
        data_LMVU=[data_LMVU;temp_list];
    end
    [m_LMVU,n_LMVU]=size(data_LMVU);
end

[m_LMVU,n_LMVU]=size(data_LMVU);

transformed_matrix=[];

for i=1:iter
    start_idx=(i-1)*batch_size+1;
    end_idx=idx(i);
    
    %Dimension Reduction using Conformal Eigenmaps
    %for every Batch
    [mappedData,mapping]= compute_mapping(data_LMVU(:,start_idx:n-1),"LandmarkMVU",2,2);

    [m_mapped,n_mapped] = size(mappedData);
    [m_transformed,n_transformed] =size(transformed_matrix);

    if(m_transformed==0)
        m_transformed=m_LMVU;
    end

    if(m_transformed>=m_mapped)
        eigen=zeros(abs(m_transformed-m_mapped),2);
        [m_mapped,n_mapped] = size([mappedData(:,:);eigen]);
        [m_transformed,n_transformed] =size(transformed_matrix);

        transformed_matrix=[transformed_matrix(:,:) [mappedData(:,:);eigen]];
    else
        
        eigen=zeros(abs(m_transformed-m_mapped),2*i);
        
        [m_mapped,n_mapped] = size(mappedData);
        [m_eigen,n_eigen]=size(eigen);
        [m_transformed,n_transformed] =size(transformed_matrix);

        transformed_matrix=[[transformed_matrix(:,:);eigen] mappedData(1:m_transformed,:)];
    end
    
    max=0;
    
    label=data_LMVU(:,n_LMVU);
end

[m_transformed,n_transformed] =size(transformed_matrix);
transformed_matrix(:,n_transformed+1)=label;

%Writing the data to a CSV
writematrix(transformed_matrix,LMVU_Dir,"Delimiter","comma");
disp("The dimension reduced (Landmark Maximum Variance Unfolding) dataset has been written to a CSV file")
