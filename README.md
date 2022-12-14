

#### 项目结构说明

```yaml
BoneRegistration/: 
	config/: #几何参数配置
	data/: #需要的数据
	calculate_parameter.py: #计算变换参数
	fit.py:
	load_xml.py:
	parse_geom.py:
	recon_4D.py: #重建入口
	util.py: 
	README.md: #项目说明
```



| file        | dims                 | brief        | dtype  | scale |
| ----------- | -------------------- | ------------ | ------ | ----- |
| 2D/         | 36 projs*(3072*3072) | 运动投影序列 | uint16 | 0.1mm |
| recon/      | 394 slices*(394*394) | CBCT重建图像 | dicom  | 0.5mm |
| 3D.raw      |                      |              |        |       |
| edge.raw    |                      |              |        |       |
| mask.raw    |                      |              |        |       |
| proc_2D.raw |                      |              |        |       |

#### 使用

##### 需要的数据：

1. 2D CBCT 投影图像序列-raw图
2. 根据投影图像提取的edge图-sobel算子提取边缘
3. 重建的xml参数文件
4. 重建3D CBCT 图像
5. 根据3D CBCT图像提取的骨骼mask图像（非骨骼区域为0，其余每块骨骼分别为1,2,3…）

 

##### 重建流程：

1. calculate_parameter.py 计算并保存变换参数
2. recon_4D.py 生成配准后的3D CT图像

 
