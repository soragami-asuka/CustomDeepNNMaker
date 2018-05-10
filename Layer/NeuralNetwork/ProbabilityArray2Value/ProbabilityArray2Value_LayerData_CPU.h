//======================================
// �o�͐M���������C���[�̃f�[�^
//======================================
#ifndef __ProbabilityArray2Value_LAYERDATA_CPU_H__
#define __ProbabilityArray2Value_LAYERDATA_CPU_H__

#include"ProbabilityArray2Value_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class ProbabilityArray2Value_LayerData_CPU : public ProbabilityArray2Value_LayerData_Base
	{
		friend class ProbabilityArray2Value_CPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		ProbabilityArray2Value_LayerData_CPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~ProbabilityArray2Value_LayerData_CPU();


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