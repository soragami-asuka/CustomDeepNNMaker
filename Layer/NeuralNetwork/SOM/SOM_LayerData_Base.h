//======================================
// �S�����j���[�����l�b�g���[�N�̃��C���[�f�[�^
//======================================
#ifndef __SOM_DATA_BASE_H__
#define __SOM_DATA_BASE_H__

#include<Layer/ILayerDataSOM.h>
#include<Layer/NeuralNetwork/IOptimizer.h>

#include<vector>

#include"SOM_DATA.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
	
	typedef F32 NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class SOM_LayerData_Base : public ILayerDataSOM
	{
	protected:
		Gravisbell::GUID guid;	/**< ���C���[�f�[�^���ʗp��GUID */

		SettingData::Standard::IData* pLayerStructure;	/**< ���C���[�\�����`�����R���t�B�O�N���X */
		SOM::LayerStructure layerStructure;	/**< ���C���[�\�� */


		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		SOM_LayerData_Base(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		virtual ~SOM_LayerData_Base();


		//===========================
		// ���ʏ���
		//===========================
	public:
		/** ���C���[�ŗL��GUID���擾���� */
		Gravisbell::GUID GetGUID(void)const;

		/** ���C���[��ʎ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		Gravisbell::GUID GetLayerCode(void)const;


		//===========================
		// ���C���[�ݒ�
		//===========================
	public:
		/** �ݒ����ݒ� */
		ErrorCode SetLayerConfig(const SettingData::Standard::IData& config);

		/** ���C���[�̐ݒ�����擾���� */
		const SettingData::Standard::IData* GetLayerStructure()const;


		//===========================
		// ���C���[�ۑ�
		//===========================
	public:
		/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
		U64 GetUseBufferByteCount()const;


	public:
		//===========================
		// ���C���[�\��
		//===========================
		/** ���̓f�[�^�\�����g�p�\���m�F����.
			@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
			@return	�g�p�\�ȓ��̓f�[�^�\���̏ꍇtrue���Ԃ�. */
		bool CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);

		/** �o�̓f�[�^�\�����擾����.
			@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
			@return	���̓f�[�^�\�����s���ȏꍇ(x=0,y=0,z=0,ch=0)���Ԃ�. */
		IODataStruct GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);

		/** �����o�͂��\�����m�F���� */
		bool CheckCanHaveMultOutputLayer(void);


		//===========================
		// �ŗL�֐�
		//===========================
	public:
		/** ���̓o�b�t�@�����擾���� */
		U32 GetInputBufferCount()const;

		/** ���j�b�g�����擾���� */
		U32 GetUnitCount()const;


		//===========================
		// �I�v�e�B�}�C�U�[�ݒ�
		//===========================
	public:
		/** �I�v�e�B�}�C�U�[��ύX���� */
		ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[]);
		/** �I�v�e�B�}�C�U�[�̃n�C�p�[�p�����[�^��ύX���� */
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value);
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value);
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[]);


		//==================================
		// SOM�֘A����
		//==================================
	public:
		/** �}�b�v�T�C�Y���擾����.
			@return	�}�b�v�̃o�b�t�@����Ԃ�. */
		U32 GetMapSize()const;

		/** �}�b�v�̃o�b�t�@���擾����.
			@param	o_lpMapBuffer	�}�b�v���i�[����z�X�g�������o�b�t�@. GetMapSize()�̖߂�l�̗v�f�����K�v. */
		virtual Gravisbell::ErrorCode GetMapBuffer(F32* o_lpMapBuffer)const = 0;
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif