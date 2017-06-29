//======================================
// �v�[�����O���C���[�̃f�[�^
//======================================
#ifndef __Residual_LAYERDATA_CPU_H__
#define __Residual_LAYERDATA_CPU_H__

#include"Residual_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Residual_LayerData_CPU : public Residual_LayerData_Base
	{
		friend class Residual_CPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		Residual_LayerData_CPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~Residual_LayerData_CPU();


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