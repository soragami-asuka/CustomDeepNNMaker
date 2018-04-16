//======================================
// �o�͐M���������C���[�̃f�[�^
//======================================
#ifndef __SignalArray2Value_LAYERDATA_GPU_H__
#define __SignalArray2Value_LAYERDATA_GPU_H__

#include"SignalArray2Value_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class SignalArray2Value_LayerData_GPU : public SignalArray2Value_LayerData_Base
	{
		friend class SignalArray2Value_GPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		SignalArray2Value_LayerData_GPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~SignalArray2Value_LayerData_GPU();


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