//======================================
// �o�͐M���������C���[�̃f�[�^
//======================================
#ifndef __RESHAPE_MIRRROX_LAYERDATA_GPU_H__
#define __RESHAPE_MIRRROX_LAYERDATA_GPU_H__

#include"Reshape_MirrorX_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Reshape_MirrorX_LayerData_GPU : public Reshape_MirrorX_LayerData_Base
	{
		friend class Reshape_MirrorX_GPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		Reshape_MirrorX_LayerData_GPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~Reshape_MirrorX_LayerData_GPU();


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