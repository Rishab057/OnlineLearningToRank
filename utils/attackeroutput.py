# -*- coding: utf-8 -*-

import json
import os
import sys
import time
from datetime import timedelta

def create_folders(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

class AttackerFileOutput(object):

    def __init__(self, output_file_path, output_header=None, close_between_writes=False,
                 also_print=False, write_date=False):
        self._output_file_path = output_file_path
        self._close_between_writes = close_between_writes
        self._also_print = also_print
        self._original_stdout = sys.stdout
        self.write_date = write_date
        create_folders(self._output_file_path)
        self._output_file = open(self._output_file_path, 'w')
        self._file_open = True
        self._new_line = True
        self._closed = False
        if not output_header is None:
            self.write(output_header)
        self._end_write()

    def _open_file(self):
        if not self._file_open:
            self._output_file = open(self._output_file_path, 'a')
            self._file_open = True

    def _close_file(self):
        self._output_file.close()
        self._file_open = False

    def _end_write(self):
        if self._close_between_writes:
            self._close_file()

    def _write_str_to_file(self, output_str):
        self._output_file.write(output_str)
        self._new_line = output_str[-1] == '\n'

    def flush(self):
        if self._also_print:
            self._original_stdout.flush()
        self._output_file.flush()

    def write(self, output, skip_write_date=False):
        assert not self._closed
        self._open_file()
        for line in output:
            if self.write_date and self._new_line and not skip_write_date:
                line = '%s: %s' % (time.strftime('%c'), str(line))
            # assert type(line) is str, 'Output element %s is not a str' % line
            self._write_str_to_file(str(line))
            if self._also_print:
                self._original_stdout.write(line)
        self._end_write()

    def close(self):
        self._close_file()
        self._closed = True
        if self._also_print:
            self._original_stdout.write('\n')


class AttackerOutput(object):

    """
    Class designed to manage the multiprocessing of simulations over multiple datasets.
    """

    def __init__(self, simulation_arguments, simulation_name, dataset, num_click_models,
                 ranker_arguments, attacker_averager):
        self._start_time = time.time()
        self.run_index = 0
        self.attacker_output_folder = simulation_arguments.attacker_folder
        self.simulation_name = simulation_name
        self.dataset_name = dataset.name
        self.attacker_averager = attacker_averager
        self.print_output = simulation_arguments.print_output
        self._expected_runs = dataset.num_runs_per_fold * dataset.num_folds * num_click_models
        self._closed = False


        self.additional_file_name = ""
        
        if "freq" in simulation_arguments.click_models[0]:
            self.additional_file_name = "_"+simulation_arguments.click_models[0]+"_"+str(simulation_arguments.n_results)+"_res_" \
                                        +str(simulation_arguments.start)+"_start_"+str(simulation_arguments.end)+"_end_"+str(simulation_arguments.mf)+"_mf_"+str(simulation_arguments.sd_const)+"_sd_" \
                                        +str(simulation_arguments.which)+"_half_"+str(simulation_arguments.n_impressions)+"_impressions"+str(ranker_arguments['learning_rate_decay'])+"_lrdecay"

        else:
            self.additional_file_name = "_"+simulation_arguments.click_models[0]+"_"+str(simulation_arguments.n_results)+"_res_"+str(simulation_arguments.start)+"_start_"+str(simulation_arguments.end)+"_end_" \
                                        +str(simulation_arguments.which)+"_half_"+str(simulation_arguments.n_impressions)+"_impressions"+str(ranker_arguments['learning_rate_decay'])+"_lrdecay"

        self.output_path = '%s/%s/%s.out' % (self.attacker_output_folder, self.dataset_name,
                                             self.simulation_name+self.additional_file_name)
        print "output path: ", self.output_path
        combined_args = {
                'simulation_arguments': vars(simulation_arguments),
                'ranker_arguments': ranker_arguments,
            }
        if self.print_output:
            output_header = json.dumps(combined_args, sort_keys=True,
                                       indent=4, separators=(',', ': '))
            self.attacker_file_output = AttackerBufferPrintOutput(output_header)
        else:
            output_header = json.dumps(combined_args, separators=(',',':'))
            self.attacker_file_output = AttackerFileOutput(self.output_path, output_header,
                                          close_between_writes=True, also_print=False,
                                          write_date=False)

    def expected_runs(self):
        return self._expected_runs

    def finished(self):
        return self._closed and self.run_index == self._expected_runs

    def write_run_output(self, run_output):
        assert not self._closed, 'Simulation Output (%s) written to after being closed.' \
            % self.output_path

        if self.print_output:
            # self.file_output.write(json.dumps(run_output, sort_keys=True,
            #                            indent=4, separators=(',', ': ')))
            self.attacker_file_output.pretty_run_write(self.run_index, run_output)
        else:
            self.attacker_file_output.write('\n%s' % json.dumps(run_output))

        self.run_index += 1
        if self.run_index >= self._expected_runs:
            self.close()

    def close(self, output_file=None):
        self.attacker_file_output.close()
        self._closed = True
        if not self.print_output:
            self.attacker_averager.create_average_file(self)


class AttackerBufferPrintOutput(object):

    def __init__(self, output_header=None):
        self._closed = False
        self._output_list = []
        if not output_header is None:
            self.write(output_header)

    def flush(self):
        pass

    def write(self, output):
        assert not self._closed
        assert type(output) is str, 'Wrong output format %s' % type(output)
        self._output_list.append(output)

    def pretty_run_write(self, run_index, run_output):
      run_details = run_output['run_details']
      run_lines = [
          "RUN: %d" % run_index,
          "DATAFOLD: %s" % run_details['data folder'],
          "CLICK MODEL: %s" % run_details['click model'],
          "RUN TIME: %s (%.02f seconds)" % (timedelta(seconds=run_details['runtime']),
                                            run_details['runtime'])
        ]
      tag = run_details['held-out data']
      for event in run_output['run_results']:
        str_line = str(event['iteration'])
        if 'display' in event:
          str_line += ' DISPLAY: %0.3f' % event['display']
        if 'heldout' in event:
          str_line += ' %s: %0.3f' % (tag, event['heldout'])
        run_lines.append(str_line)
      for line in run_lines:
        self.write(line)

    def close(self):
        self._closed = True
        print 'Run Output\n' + '\n'.join(self._output_list)
        self._output_list = []
