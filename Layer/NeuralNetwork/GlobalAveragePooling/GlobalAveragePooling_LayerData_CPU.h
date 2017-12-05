//======================================
// �v�[�����O���C���[�̃f�[�^
//======================================
#ifndef __GlobalAveragePooling_LAYERDATA_CPU_H__
#define __GlobalAveragePooling_LAYERDATA_CPU_H__

#include"GlobalAveragePooling_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class GlobalAveragePooling_LayerData_CPU : public GlobalAveragePooling_LayerData_Base
	{
		friend class GlobalAveragePooling_CPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		GlobalAveragePooling_LayerData_CPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~GlobalAveragePooling_LayerData_CPU();


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