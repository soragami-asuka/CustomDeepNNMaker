//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̃��C���[�f�[�^
//======================================
#ifndef __CONVOLUTION_LAYERDATA_GPU_H__
#define __CONVOLUTION_LAYERDATA_GPU_H__


#include"Convolution_LayerData_Base.h"

#pragma warning(push)
#pragma warning(disable : 4267)
#pragma warning(disable : 4819)
#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#pragma warning(pop)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Convolution_LayerData_GPU : public Convolution_LayerData_Base
	{
		friend class Convolution_GPU;

	private:
		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		Convolution_LayerData_GPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~Convolution_LayerData_GPU();


		//===========================
		// ������
		//===========================
	public:
		using Convolution_LayerData_Base::Initialize;

		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@return	���������ꍇ0 */
		ErrorCode Initialize(void);


		//===========================
		// ���C���[�ۑ�
		//===========================
	public:

		//===========================
		// ���C���[�쐬
		//===========================
	public:
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);


		//===========================
		// �I�v�e�B�}�C�U�[�ݒ�
		//===========================
	public:
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif