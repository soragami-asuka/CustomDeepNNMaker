//======================================
// �S�����j���[�����l�b�g���[�N�̃��C���[�f�[�^
//======================================
#ifndef __FullyConnect_LAYERDATA_CPU_H__
#define __FullyConnect_LAYERDATA_CPU_H__

#include"FullyConnect_LayerData_Base.h"


#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#pragma warning(pop)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class FullyConnect_LayerData_GPU : public FullyConnect_LayerData_Base
	{
		friend class FullyConnect_GPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		FullyConnect_LayerData_GPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~FullyConnect_LayerData_GPU();


		//===========================
		// ������
		//===========================
	public:
		using FullyConnect_LayerData_Base::Initialize;

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

	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif