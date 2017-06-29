//======================================
// �v�[�����O���C���[�̃f�[�^
//======================================
#ifndef __GlobalAveragePooling_LAYERDATA_GPU_H__
#define __GlobalAveragePooling_LAYERDATA_GPU_H__

#include"GlobalAveragePooling_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class GlobalAveragePooling_LayerData_GPU : public GlobalAveragePooling_LayerData_Base
	{
		friend class GlobalAveragePooling_GPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		GlobalAveragePooling_LayerData_GPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~GlobalAveragePooling_LayerData_GPU();


		//===========================
		// ���C���[�쐬
		//===========================
	public:
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif