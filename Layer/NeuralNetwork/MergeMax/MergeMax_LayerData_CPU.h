//======================================
// �v�[�����O���C���[�̃f�[�^
//======================================
#ifndef __MergeMax_LAYERDATA_CPU_H__
#define __MergeMax_LAYERDATA_CPU_H__

#include"MergeMax_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class MergeMax_LayerData_CPU : public MergeMax_LayerData_Base
	{
		friend class MergeMax_CPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		MergeMax_LayerData_CPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~MergeMax_LayerData_CPU();


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